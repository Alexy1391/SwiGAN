"""Anchor the settled rollout to the training climatology, using no observation of the predictand.

WHY THIS EXISTS. ``utils/bias_correction.py`` estimated a global offset on a calibration window and
scored the transfer to the test window. It failed, identically on every checkpoint: the residual
val->test gap is +0.32 to +0.43 sd on all seven, oper->test +0.41 to +0.51, as large as the bias
being corrected. It helped the five seed-60 checkpoints and DAMAGED both seed-61 ones -- CRPS
+20.3% on the s61 baseline against a -3.7% oracle, because seed 61 has essentially no bias on test
and the validation offset injects one. ``utils/bias_structure.py`` then measured why:

    bias(window)  =  climate_anom(window)  -  model_offset(weights)

    model offset    where the free-running rollout SETTLES, measured against the 1960-2019
                    observed climatology it was trained on. A property of the WEIGHTS. Range
                    across the three windows is 0.017-0.117 sd per run: it transfers.
    climate anomaly how far the observations over the scored window sit from that same
                    climatology. A property of the WEATHER: +0.055 -> -0.543 -> +0.442 across the
                    three windows, identical for every model, and contained in no window that
                    ends before the forecast starts.

Fitting one scalar to the sum fits the half that does not transfer. This script removes the half
that does, and nothing else.

THE CORRECTION. Shift every member by ``-w(k) * offset[month(k)]``, where ``offset`` is the model's
own settled departure from climatology per calendar month and ``w`` ramps the shift in as the
initial condition decays. By construction the corrected ensemble mean's settled seasonal cycle IS
the 1960-2019 observed climatology -- that is the anchor, and it is checked numerically below
rather than asserted. What is left at long lead is then exactly the climate anomaly, which is a
training-distribution problem (retrain window, trend term), not a calibration one.

WHY THIS IS NOT THE SAME MOVE AGAIN. ``b_mean`` and ``b_crps`` need ``y`` over a calibration
window; the estimate is only as good as that window's weather, which is what broke. The model
offset is ``rollout_mean - climatology_1960_2019``, and NEITHER TERM SEES ``swi_u`` OVER THE SCORED
WINDOW. The first is the model's own forecast, driven by the same bias-adjusted historical/rcp85
scenario forcings the forecast itself runs on -- those are projection inputs the system has in hand
by construction, not observations of anything. The second is a fixed 1960-2019 climatology of the
predictand, taken from inside the training split. So the offset is legitimately estimable ON THE
SCORED ROLLOUT ITSELF, at forecast issue time, with nothing in hand but the weights, the forcings
and the climatology -- the ``self`` source below. That is not leakage and it is not a free parameter
fitted to the test set; it is a quantity the forecast system can compute before the forecast
verifies. The ``val`` and ``oper`` sources estimate the same thing from a rollout that ends before
the forecast starts, and are scored alongside so the transfer claim is measured rather than argued:
if ``self`` and ``oper`` agree, the estimate never needed the scored window at all.

WHY YEAR 2 ONLY, AND WHY THE RAMP. ``pred(k) - climatology`` at short lead is mostly SIGNAL: the
rollout still remembers the initial condition and is supposed to depart from climatology. Removing
all of it would not debias the forecast, it would replace the forecast with climatology. Only once
the initial condition has decayed is the whole departure drift. So the offset is measured on year 2
alone, and applied through a weight that grows from 0 at initialisation to 1 by lead 13:

    ramp     w = min(k/12, 1)          drift grows as the IC decays. The a-priori default.
    settled  w = 1 for k >= 12, else 0  applied only where it was measured. Discontinuous.
    full     w = 1 everywhere           over-application, scored to show what that costs.

The three profiles are fixed here in advance and all three are reported. None is selected on the
test score; picking the best of them by CRPS on the scored window would be the same mistake in a
new costume.

SELF-LIMITING, WHICH IS THE POINT. Seed 61's model offset is +0.104 / +0.064 / -0.013 sd across the
three windows -- essentially zero -- so this correction correctly applies almost nothing to it and
cannot damage it the way the validation offset did. Seed 60's is -0.20 to -0.44 sd, and there it
does real work: roughly half the year-2 bias on the r8 baseline. The correction's size is set by
how far the model drifts, not by how odd the calibration window's weather was.

WHAT IT DOES NOT FIX. Dispersion. Debiasing exposes it: at the oracle the s60 baseline rank
histogram flips from right-skewed to a clean U with 71.5% of observations outside the envelope.
That is the next lever, not this one. The shift is also spatially constant -- one masked-mean
number per calendar month, matching the global-offset work it replaces -- so any per-pixel
structure in the drift is untouched here.
"""

import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.bias_correction import WINDOWS, offsets, rmse_terms, rollout
from utils.bias_structure import MONTHS, TRAIN_END_YEAR, observed_climatology, window_months
from utils.ensemble_scores import score

logger = logging.getLogger(__file__)

SETTLED_FROM = 12  # first rollout step treated as settled; year 2 of a 24-step window
WEIGHT_PROFILES = ("ramp", "settled", "full")


def settled_offset(pred: np.ndarray, months: list[tuple[int, int]], climatology: pd.Series,
                   settled_from: int = SETTLED_FROM) -> np.ndarray:
    """Model offset per calendar month, from the settled part of one rollout.

    ``pred`` is the masked-mean of the ensemble mean at each rollout step, (T,). Steps before
    ``settled_from`` are dropped: there the departure from climatology is the forecast doing its
    job, not drift. Returns (12,) indexed by calendar month - 1, NaN for months the settled part
    never reaches. No observation of the predictand over the rollout window enters this.
    """
    total, count = np.zeros(12), np.zeros(12)
    for k in range(settled_from, len(pred)):
        month = months[k][1]
        total[month - 1] += pred[k] - climatology[month]
        count[month - 1] += 1
    return np.where(count > 0, total / np.maximum(count, 1.0), np.nan)


def weight_profile(name: str, steps: int, settled_from: int = SETTLED_FROM) -> np.ndarray:
    """How much of the settled offset to apply at each rollout step, (T,) in [0, 1]."""
    k = np.arange(steps, dtype=float)
    if name == "ramp":
        return np.clip(k / settled_from, 0.0, 1.0)
    if name == "settled":
        return (k >= settled_from).astype(float)
    if name == "full":
        return np.ones(steps)
    raise ValueError(f"unknown weight profile {name!r}")


def anchor_shift(offset_by_month: np.ndarray, months: list[tuple[int, int]], profile: str,
                 seasonal: bool) -> np.ndarray:
    """The per-step amount to ADD to every member, (T,), in raw swi_u units.

    ``seasonal`` keeps the twelve monthly values; otherwise their mean is used at every step, which
    is the one-scalar form of the same correction. A month the estimate never reached falls back to
    that mean rather than propagating a NaN.
    """
    mean_offset = float(np.nanmean(offset_by_month))
    if seasonal:
        per_month = np.where(np.isnan(offset_by_month), mean_offset, offset_by_month)
        per_step = np.array([per_month[month - 1] for _year, month in months])
    else:
        per_step = np.full(len(months), mean_offset)
    return -weight_profile(profile, len(months)) * per_step


def score_anchored(members: np.ndarray, truth: np.ndarray, target_std: float,
                   shift: np.ndarray) -> dict:
    """Every score for the ensemble shifted by a per-step ``shift`` (T,), plus its residual bias."""
    shifted = members + shift[:, None]
    out = score(shifted, truth, target_std)
    out.update(rmse_terms(shifted, truth))
    residual = truth - shifted.mean(axis=0)  # (T, P)
    out["bias_sd"] = float(residual.mean()) / target_std
    # The settled-only residual is the one with a predicted value: once the anchor is applied at
    # full weight the corrected mean IS climatology, so what is left here must be the window's
    # climate anomaly exactly. The 24-month mean mixes that with year 1, where the ramp is only
    # partly on, and so converges to nothing in particular.
    out["bias_y2_sd"] = float(residual[SETTLED_FROM:].mean()) / target_std
    out["offset"] = float(shift.mean())
    out["offset_y2_sd"] = float(shift[SETTLED_FROM:].mean()) / target_std
    return out


def climate_anomaly(truth: np.ndarray, months: list[tuple[int, int]], climatology: pd.Series,
                    settled_from: int = SETTLED_FROM) -> float:
    """How far the observations over the settled part sit from climatology, in raw swi_u units.

    This is the half of the bias no calibration can reach. It is a property of the window's
    weather, so it is the same number for every checkpoint, and it is what the anchored residual
    should converge to if the anchor is doing what it claims.
    """
    observed = truth.mean(axis=1)[settled_from:]
    reference = np.array([climatology[month] for _year, month in months[settled_from:]])
    return float((observed - reference).mean())


def check_anchor(pred: np.ndarray, months: list[tuple[int, int]], climatology: pd.Series,
                 offset_by_month: np.ndarray, target_std: float) -> list[str]:
    """Verify the anchor does what it claims: settled cycle onto climatology, nothing before it.

    The seasonal/settled form is the one with an exact property to check -- after it, the corrected
    ensemble mean over year 2 must equal the 1960-2019 climatology month by month, to machine
    precision, because that is the definition of the shift and not a fitted outcome. The ramp form
    is checked more loosely for the property that actually matters at short lead: it must leave the
    first month essentially untouched.
    """
    notes = []
    shift = anchor_shift(offset_by_month, months, "settled", seasonal=True)
    total, count = np.zeros(12), np.zeros(12)
    for k in range(SETTLED_FROM, len(pred)):
        month = months[k][1]
        total[month - 1] += pred[k] + shift[k] - climatology[month]
        count[month - 1] += 1
    worst = float(np.abs(total[count > 0] / count[count > 0]).max())
    notes.append(
        f"seasonal/settled puts the year-2 cycle exactly on climatology: worst month "
        f"{worst:.2e} ({'OK' if worst < 1e-6 else 'FAILED'})"
    )

    ramp = anchor_shift(offset_by_month, months, "ramp", seasonal=True)
    lead1 = abs(float(ramp[0])) / target_std
    notes.append(
        f"ramp leaves lead 1 to the initial condition: shift {lead1:.2e} sd "
        f"({'OK' if lead1 < 1e-9 else 'FAILED'})"
    )

    tail = np.abs(ramp[SETTLED_FROM:] - shift[SETTLED_FROM:]).max() / target_std
    notes.append(
        f"ramp and settled agree once settled: worst gap {tail:.2e} sd "
        f"({'OK' if tail < 1e-9 else 'FAILED'})"
    )
    return notes


ROW = ("{:<30s} {:>8.5f} {:>9.5f} {:>+8.3f} {:>11.3f} {:>8.3f} {:>7.1f}% "
       "{:>9.4f} {:>9.4f} {:>+8.3f} {:>+9.3f}")
HEAD = ("{:<30s} {:>8s} {:>9s} {:>8s} {:>11s} {:>8s} {:>8s} {:>9s} {:>9s} {:>8s} {:>9s}").format(
    "arm", "CRPS", "CRPS det", "skill", "sprd/skill", "reliab", "outside", "RMSE ens", "RMSE mem",
    "bias sd", "shift y2")


def build_arms(offset: dict[str, np.ndarray], months: list[tuple[int, int]],
               b_val: dict[str, float], b_test: dict[str, float],
               steps: int) -> list[tuple[str, np.ndarray]]:
    """The scored arms, as (label, per-step shift). Fixed in advance, not chosen by the score.

    ``self`` is the operational form: the offset read off the very rollout being corrected, which
    it may do because no observation enters the estimate. ``oper`` and ``val`` re-estimate the same
    quantity from an older rollout and exist to show how little it moves. The two reference rows
    are the failed constant from ``bias_correction.py`` and ``b_oracle``, so this table can be read
    directly against that one.

    ``b_oracle`` is the best CONSTANT fitted to the test window, which is the ceiling for the
    family ``bias_correction.py`` searched and NOT a ceiling for this one. An anchored arm may
    legitimately beat it: the shift here varies with lead time and season, so it can correct the
    settled tail without dragging the initial-condition-driven months off with it, which no single
    constant can do. Read it as "the most a constant could ever buy", not "the most anything can".
    """
    flat = np.ones(steps)
    arms: list[tuple[str, np.ndarray]] = [
        ("uncorrected", np.zeros(steps)),
        ("+ b_crps   val, ref", flat * b_val["b_crps"]),
    ]
    for profile in WEIGHT_PROFILES:
        arms.append((f"anchor self  seas  {profile}",
                     anchor_shift(offset["test"], months, profile, seasonal=True)))
    arms.append(("anchor self  const ramp",
                 anchor_shift(offset["test"], months, "ramp", seasonal=False)))
    for source in ("oper", "val"):
        arms.append((f"anchor {source:5s} seas  ramp",
                     anchor_shift(offset[source], months, "ramp", seasonal=True)))
    arms.append(("+ b_oracle test, best const", flat * b_test["b_mean"]))
    return arms


def report_offsets(offset: dict[str, np.ndarray], b: dict[str, dict[str, float]],
                   target_std: float) -> None:
    """Print the settled offset per window, against what ``bias_correction.py`` estimated instead.

    The two gap lines are the whole argument side by side, computed on the very same rollouts:
    what this estimator fails to carry from one window to the next, and what ``b_mean`` fails to
    carry. They differ by an order of magnitude, and the reason is that only one of them contains
    the window's weather.
    """
    print(f"\n  THE SETTLED MODEL OFFSET, by calendar month, in sd of targets_std "
          f"{target_std:.4f}   (rollout year 2 minus climatology; no swi_u from the window)")
    print(f"  {'estimated on':22s}" + "".join(f"{m:>7s}" for m in MONTHS) + f"{'mean':>9s}")
    print("  " + "-" * 115)
    for w, (label, *_rest) in WINDOWS.items():
        row = offset[w] / target_std
        print(f"  {label:22s}" + "".join(f"{v:+7.3f}" for v in row) + f"{np.nanmean(row):+9.3f}")
    print("  " + "-" * 115)
    means = {w: float(np.nanmean(offset[w])) for w in WINDOWS}
    spread = (max(means.values()) - min(means.values())) / target_std
    print(f"  {'THIS estimator, gap to test':26s} "
          f"val {(means['test'] - means['val']) / target_std:+7.3f} sd    "
          f"oper {(means['test'] - means['oper']) / target_std:+7.3f} sd    "
          f"full range {spread:6.3f} sd")
    print(f"  {'b_mean, same comparison':26s} "
          f"val {(b['test']['b_mean'] - b['val']['b_mean']) / target_std:+7.3f} sd    "
          f"oper {(b['test']['b_mean'] - b['oper']['b_mean']) / target_std:+7.3f} sd")
    print("  b_mean needs the observations over its window and inherits that window's weather.")
    print("  This one needs only the weights and a fixed 1960-2019 climatology, so what it")
    print("  estimates is the model's drift and it holds still. Run-to-run rollout noise is")
    print("  ~0.001 sd on this mean (fp16 inference), well inside the spread above.")


def report_windows(runs: dict, months: dict, offset: dict, anom: dict[str, float], key: str,
                   target_std: float) -> dict[str, tuple[dict, dict]]:
    """Score the same correction on all three windows, and say what that does and does not show.

    The test window's climate anomaly (+0.44 sd) and seed 60's drift (-0.44 sd) happen to push the
    same way, so on that window alone removing the drift looks like a large unqualified win.
    Scoring every window is what separates the correction from that coincidence: ``oper`` has an
    anomaly of -0.54 sd, where the drift was CANCELLING the error rather than adding to it, and
    anchoring there must be expected to hurt. Reported because it is the honest shape of the
    result, not despite it.

    Each row uses the offset taken from the window being scored, so every row is the operational
    protocol rather than a transfer test. Returns the raw and anchored scores per window.
    """
    print("\n  THE SAME CORRECTION ON ALL THREE WINDOWS  (anchor self seas ramp, offset always")
    print("  taken from the window being scored, so every row is the operational protocol)")
    print(f"  {'scored window':22s} {'clim anom':>10s} | {'y2 bias raw':>11s} {'y2 anch':>9s} "
          f"{'24m raw':>8s} {'24m anch':>9s} | {'CRPS raw':>9s} {'CRPS anch':>10s} {'CRPS':>7s} "
          f"{'RMSE':>7s}")
    print("  " + "-" * 112)
    window_rows, landed = {}, []
    for w, (label, *_rest) in WINDOWS.items():
        members, truth = runs[w][key], runs[w]["truth"]
        shift = anchor_shift(offset[w], months[w], "ramp", seasonal=True)
        raw = score_anchored(members, truth, target_std, np.zeros(len(months[w])))
        anchored = score_anchored(members, truth, target_std, shift)
        window_rows[w] = (raw, anchored)
        landed.append(abs(anchored["bias_y2_sd"] - anom[w]))
        print(f"  {label:22s} {anom[w]:+10.3f} | {raw['bias_y2_sd']:+11.3f} "
              f"{anchored['bias_y2_sd']:+9.3f} {raw['bias_sd']:+8.3f} {anchored['bias_sd']:+9.3f} "
              f"| {raw['crps']:9.5f} {anchored['crps']:10.5f} "
              f"{(anchored['crps'] / raw['crps'] - 1) * 100:+6.1f}% "
              f"{(anchored['rmse_ens'] / raw['rmse_ens'] - 1) * 100:+6.1f}%")
    print("  " + "-" * 112)
    worst = max(landed)
    print(f"  check  the anchored YEAR-2 residual equals the climate anomaly: worst gap "
          f"{worst:.2e} sd ({'OK' if worst < 1e-6 else 'FAILED'})")
    print("  That identity is the correction landing exactly where it was aimed, and it is not")
    print("  fitted: at full weight the corrected settled mean IS climatology, so what is left")
    print("  must be the window's departure from it. Every checkpoint therefore ends on the SAME")
    print("  three y2 residuals -- which is the proof the removed part was the model's and the")
    print("  remaining part never was. The 24m columns mix year 2 with year 1, where the ramp is")
    print("  only partly on, so they converge to nothing in particular; read y2.")
    print("")
    print("  It is NOT a uniform CRPS win, and that is the honest shape of the result. On test the")
    print("  weather sat on the same side of climatology as the drift, so removing the drift moves")
    print(f"  toward the truth. On oper the anomaly is {anom['oper']:+.3f} sd and the drift was")
    print("  CANCELLING it, so removing it moves away and CRPS rises. The mean anomaly over the")
    print(f"  three windows is {np.mean(list(anom.values())):+.3f} sd, near enough zero that the")
    print("  drift is the systematic part and removing it is the right call in expectation. A")
    print("  drift is real; a cancellation is a coincidence. Fitting the cancellation is what")
    print("  b_mean did, and why it inverted between seeds.")
    return window_rows


def collect(saved: dict, key: str, arms: list, rows: list, offset: dict, pred: dict,
            window_rows: dict, anom: dict, members: np.ndarray, truth: np.ndarray,
            target_std: float) -> None:
    """Put everything the driver's comparison needs into ``saved``, keyed by dropout arm."""
    saved[f"{key}_target_std"] = np.array(target_std)
    saved[f"{key}_arms"] = np.array([label for label, _shift in arms])
    saved[f"{key}_shift"] = np.stack([shift for _label, shift in arms])
    for w in WINDOWS:
        saved[f"{key}_{w}_offset_by_month"] = offset[w]
        saved[f"{key}_{w}_settled_pred"] = pred[w][SETTLED_FROM:]
        saved[f"{key}_{w}_clim_anom"] = np.array(anom[w])
        for name, s in zip(("raw", "anch"), window_rows[w], strict=True):
            for field in ("crps", "bias_sd", "bias_y2_sd", "rmse_ens", "spread_skill",
                          "extreme_frac"):
                saved[f"{key}_{w}_{name}_{field}"] = np.array(s[field])
    for field in ("crps", "crps_det", "crps_skill", "spread_skill", "reliability", "extreme_frac",
                  "coverage", "rmse_ens", "rmse_mem", "rmse_max", "bias_sd", "bias_y2_sd",
                  "offset", "offset_y2_sd"):
        saved[f"{key}_{field}"] = np.array([s[field] for _label, s in rows])
    saved[f"{key}_rank_hist"] = np.stack([s["rank_hist"] for _label, s in rows])
    saved[f"{key}_resid_by_lead"] = np.stack([
        (truth - (members + shift[:, None]).mean(axis=0)).mean(axis=1) / target_std
        for _label, shift in arms])


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Estimate the settled model offset, anchor the rollout to climatology, score on test."""
    climatology, _observed = observed_climatology(cfg)
    runs = {key: rollout(cfg, year, month, steps)
            for key, (_label, year, month, steps) in WINDOWS.items()}
    months = {key: window_months(year, month, steps)
              for key, (_label, year, month, steps) in WINDOWS.items()}

    saved: dict[str, np.ndarray] = {}
    for key in ("off", "on"):
        arm = "dropout ON " if key == "on" else "dropout off"
        target_std = runs["test"]["target_std"]
        test_members, test_truth = runs["test"][key], runs["test"]["truth"]
        steps = test_members.shape[1]

        pred = {w: runs[w][key].mean(axis=0).mean(axis=1) for w in WINDOWS}  # (T,) per window
        offset = {w: settled_offset(pred[w], months[w], climatology) for w in WINDOWS}
        anom_by_window = {w: climate_anomaly(runs[w]["truth"], months[w], climatology) / target_std
                          for w in WINDOWS}
        b = {w: offsets(runs[w][key], runs[w]["truth"]) for w in WINDOWS}
        notes = check_anchor(pred["test"], months["test"], climatology, offset["test"], target_std)

        arms = build_arms(offset, months["test"], b["val"], b["test"], steps)
        rows = [(label, score_anchored(test_members, test_truth, target_std, shift))
                for label, shift in arms]

        print("\n" + "=" * 133)
        print(f"CLIMATOLOGY-ANCHORED CORRECTION  ·  {arm}".center(133))
        print(f"settled rollout pinned to the 1960-{TRAIN_END_YEAR} observed climatology · "
              f"scored on {WINDOWS['test'][0]}".center(133))
        print("=" * 133)
        for note in notes:
            print(f"  check  {note}")

        report_offsets(offset, b, target_std)

        print(f"\n  SCORED ON THE TEST WINDOW  {WINDOWS['test'][0]}")
        print("  " + HEAD)
        print("  " + "-" * 131)
        base = rows[0][1]
        for label, s in rows:
            print("  " + ROW.format(label, s["crps"], s["crps_det"], s["crps_skill"],
                                    s["spread_skill"], s["reliability"], s["extreme_frac"] * 100,
                                    s["rmse_ens"], s["rmse_mem"], s["bias_sd"], s["offset_y2_sd"]))
        print("  " + "-" * 131)
        for label, s in rows[1:]:
            hist, base_hist = s["rank_hist"], base["rank_hist"]
            print(f"  {label:<28s} CRPS {(s['crps'] / base['crps'] - 1) * 100:+6.1f}%   "
                  f"RMSE ens {(s['rmse_ens'] / base['rmse_ens'] - 1) * 100:+6.1f}%   "
                  f"reliab {base['reliability']:.3f}->{s['reliability']:.3f}   "
                  f"bin 0 {base_hist[0] * 100:4.1f}->{hist[0] * 100:4.1f}%   "
                  f"bin 20 {base_hist[-1] * 100:4.1f}->{hist[-1] * 100:4.1f}%")

        print("\n  WHAT IS LEFT, by lead-time block, in sd  (mean of y - corrected ensemble mean)")
        print(f"  {'arm':30s} {'m1-6':>9s} {'m7-12':>9s} {'m13-18':>9s} {'m19-24':>9s} "
              f"{'24m':>9s}")
        print("  " + "-" * 82)
        for label, shift in arms:
            resid = (test_truth - (test_members + shift[:, None]).mean(axis=0))
            by_lead = resid.mean(axis=1) / target_std
            print(f"  {label:30s} {by_lead[:6].mean():+9.3f} {by_lead[6:12].mean():+9.3f} "
                  f"{by_lead[12:18].mean():+9.3f} {by_lead[18:].mean():+9.3f} "
                  f"{by_lead.mean():+9.3f}")
        print("  " + "-" * 82)
        print("  The year-2 residual of the anchored arms is the CLIMATE ANOMALY of this window,")
        print(f"  {anom_by_window['test']:+.3f} sd. It is identical for every model and no")
        print("  pre-forecast window holds it. Removing it is a training-distribution question,")
        print("  not a calibration one.")

        window_rows = report_windows(runs, months, offset, anom_by_window, key, target_std)
        print("=" * 133)

        collect(saved, key, arms, rows, offset, pred, window_rows, anom_by_window,
                test_members, test_truth, target_std)

    saved["climatology"] = climatology.to_numpy()
    out = cfg["model"].get("saving_path")
    if out:
        np.savez(f"{out}_anchor.npz", **saved)
        logger.info(f"Saved to '{out}_anchor.npz'.")


if __name__ == "__main__":
    main()
