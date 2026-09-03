"""Choose the injection gain on calibration windows only, and score the transfer to test.

WHY THIS EXISTS. ``utils/noise_gain_sweep.py`` swept the gain ON the scored window and reported
where CRPS bottomed out: g = 2.5 on the seed-60 round-8 baseline, worth -18.3%. That number is an
UPPER BOUND, not a result, for exactly the reason ``utils/bias_correction.py``'s -23.1% was: the
parameter was chosen by the score it is quoted against. The anchor's offset escaped that trap
because it needs no observation of the predictand and can be computed at issue time. A GAIN CHOSEN
BY CRPS DOES NOT ESCAPE IT -- CRPS needs ``y``. So the gain has to be picked on a window that ends
before the forecast starts, and the transfer has to be measured rather than assumed.

THE PREDICTION, MADE BEFORE THE MEASUREMENT. There is a specific reason to expect single-window
selection to transfer badly, and it is the same mechanism that broke ``b_mean``. At long lead the
error of the anchored ensemble mean IS the window's climate anomaly, and CRPS balances spread
against that error. So the gain a window asks for scales with |anomaly| of that window:

    val   2020-01..2021-12   year-2 anomaly  +0.055 sd   nearly none -> asks for LITTLE spread
    oper  2021-01..2022-12   year-2 anomaly  -0.543 sd   large       -> asks for MUCH spread
    test  2023-01..2024-12   year-2 anomaly  +0.442 sd   large       -> needs MUCH spread

Which predicts, in advance: **val under-selects the gain and oper roughly matches test, and neither
because of anything about the model.** Val is the split the model was validated on and the window
the question names, and it is the one with almost no weather in it. If that prediction holds, the
lesson is not "use oper instead" -- oper agreeing with test is luck of the same kind that made
b_mean look good on seed 60 -- it is that ONE window cannot set this parameter, and the estimate
has to be pooled over enough windows that the anomaly averages out. Hence ``pool`` below, and hence
``clim``, which never looks at ``y`` at all.

THE RULES. All are fixed here in advance and every one is scored on test. None is selected by its
test score; ``oracle`` is reported beside them as the ceiling, and is not a candidate.

  val        CRPS-argmin on the 2020-01 window alone. The literal reading of "choose it on val".
  oper       CRPS-argmin on the 2021-01 window alone -- the 24 months actually in hand at issue.
  pool post  CRPS-argmin on the mean CRPS over the windows not fully inside the training split.
  pool all   CRPS-argmin on the mean CRPS over every calibration window.
  clim       the gain whose settled spread matches the climatological sd of the 12-month observed
             anomaly (0.307 sd over 65 January-start blocks). Uses NO observation of y over any
             window -- only the model's own spread and a training-era climatology -- so it is the
             one rule that is available at issue time in the same sense the anchor's offset is.
  oracle     CRPS-argmin on test. THE CEILING. Quoted so the transfer loss is visible.

LEAD DEPENDENCE, AND WHY IT NEEDS A SCHEDULE. A rollout is autoregressive and uses one gain for all
24 steps, so "a gain that varies with lead time" is not a post-hoc reweighting -- it has to be
modulated inside the loop. ``gain_schedule`` does that by scaling ``noise_weights`` before each
forward, which reuses ``compute_trajectory_iterative`` unchanged rather than reimplementing it. A
constant schedule reproduces ``noise_gain`` BIT-EXACTLY, which is checked below and is what makes
the duplication safe. Two families are scored:

    const  g(k) = g                                  one number, what the sweep measured
    ramp   g(k) = 1 + (g - 1) * min(k / 12, 1)       trained operating point at short lead, g by
                                                     year 2 -- the same shape as the anchor's ramp

The ramp exists because the sweep showed the two ends of the lead time want different gains: at
m1-12 the initial condition is live and the ensemble should stay narrow, at m13-24 it has to cover
the anomaly. Note the two families are NOT independent knobs per lead -- raising the gain early
propagates through the rollout -- so this is a one-parameter family with a fixed shape, not a free
per-lead table. That is deliberate: a free table would be several more fitted parameters.

WHY ALSO ON WINDOWS INSIDE THE TRAINING SPLIT. Most calibration windows here start before 2019, so
the weights saw those targets. That should make the rollout look BETTER there than it will on test,
which would bias the pooled gain DOWN. It is included and flagged rather than dropped, because the
alternative is estimating from the three post-training windows alone, and three is the sample size
that produced the 0.496 sd target the sweep had to correct. Both pools are reported.
"""

import copy
import itertools
import logging
from contextlib import contextmanager

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from utils.bias_correction import WINDOWS, rmse_terms
from utils.bias_structure import observed_climatology, window_months
from utils.climatology_anchor import SETTLED_FROM, anchor_shift, settled_offset
from utils.dropout_probe import dropout_active
from utils.ensemble_scores import score
from utils.noise_gain_sweep import (
    GAINS,
    REFERENCE_GAIN,
    anomaly_targets,
    crossing,
    spread_sd,
    weights_rms,
)
from utils.stochasticity_probe import prepare, run_arm

logger = logging.getLogger(__file__)

STEPS = 24
TEST_YEAR = 2023
# January starts, 24 steps, every one ending before the test window opens. All start in January so
# lead time lands on the same calendar month in every window, as WINDOWS already requires.
CALIB_YEARS: tuple[int, ...] = tuple(range(2008, 2022))
POST_TRAIN_FROM = 2019  # windows from here on are not fully inside the 1960-01..2019-09 train split
FAMILIES = ("const", "ramp")
SELECT_BLOCK = "m1-24"  # the whole forecast is the deployable object, so it is the objective
BLOCKS: tuple[tuple[str, slice], ...] = (
    ("m1-24", slice(0, 24)),
    ("m1-12", slice(0, 12)),
    ("m13-24", slice(12, 24)),
)
DROPOUT = True


@contextmanager
def gain_schedule(model: torch.nn.Module, schedule: np.ndarray):
    """Apply a per-rollout-step gain to ``noise_weights``, one value per autoregressive step.

    ``compute_trajectory_iterative`` calls the model once per step, so wrapping ``forward`` with a
    step counter is enough to make the gain lead-dependent WITHOUT reimplementing the rollout --
    the same monkeypatch idiom ``stochasticity_probe.shared_injection`` already uses. Scaling the
    weights consumes no random numbers, so the RNG stream is untouched and a constant schedule is
    bit-identical to ``noise_gain`` at that value. ``check_schedule`` asserts exactly that.

    The counter is clamped at the end of the schedule so a shorter schedule holds its last value,
    and the weights are restored from the saved clone rather than divided back, so a long sweep
    cannot drift the checkpoint under itself.
    """
    named = [(name, p) for name, p in model.named_parameters() if "noise_weights" in name]
    if not named:
        raise RuntimeError("no noise_weights parameters found -- the model is not the SwiGAN UNet")
    base = {name: p.detach().clone() for name, p in named}
    original = model.forward
    counter = itertools.count()

    def patched(*args, **kwargs):
        gain = float(schedule[min(next(counter), len(schedule) - 1)])
        with torch.no_grad():
            for name, p in named:
                p.copy_(base[name] * gain)
        return original(*args, **kwargs)

    model.forward = patched
    try:
        yield
    finally:
        del model.forward  # drop the instance attribute, exposing the bound method again
        with torch.no_grad():
            for name, p in named:
                p.copy_(base[name])


def build_schedule(family: str, gain: float, steps: int = STEPS) -> np.ndarray:
    """The per-step gain for one family. ``const`` is flat; ``ramp`` mirrors the anchor's ramp."""
    if family == "const":
        return np.full(steps, gain, dtype=float)
    if family == "ramp":
        return 1.0 + (gain - 1.0) * np.clip(np.arange(steps) / SETTLED_FROM, 0.0, 1.0)
    raise ValueError(f"unknown family {family!r}")


def check_schedule(model: torch.nn.Module, shared: dict, bool_mask: np.ndarray,
                   target_std: float) -> list[str]:
    """Verify the schedule mechanism before any selection is read off it.

    Four checks, in the order they would fail: the lever exists; a flat schedule is the sweep's own
    ``noise_gain`` bit for bit (which is what licenses reusing that round's numbers as the const
    family); the ramp actually separates the two ends of the lead time in the ROLLOUT rather than
    only on paper; and the weights come back untouched, so 360 schedules cannot drift the checkpoint
    under itself.
    """
    from utils.noise_gain_sweep import noise_gain

    named = [(n, p) for n, p in model.named_parameters() if "noise_weights" in n]
    before = {n: p.detach().clone() for n, p in named}
    notes = [f"the schedule reaches {len(named)} noise_weights tensors, "
             f"rms {weights_rms(model):.6f}"]
    gain = 2.5

    with noise_gain(model, gain), dropout_active(model, DROPOUT):
        direct = run_arm(noise_std=1.0, injection="independent", seed=0, **shared)
    with gain_schedule(model, build_schedule("const", gain)), dropout_active(model, DROPOUT):
        via = run_arm(noise_std=1.0, injection="independent", seed=0, **shared)
    worst = float(np.abs(direct - via).max())
    notes.append(
        f"a constant schedule reproduces noise_gain bit-exactly at g={gain}: max |diff| "
        f"{worst:.2e} ({'OK' if worst == 0.0 else 'FAILED'})"
    )

    # The shape, on paper. A ramp that reached g at lead 1 would be a constant under another name.
    ramp = build_schedule("ramp", gain)
    notes.append(
        f"the ramp leaves lead 1 at the trained operating point and reaches g by lead 13: "
        f"g(0)={ramp[0]:.3f}, g(12)={ramp[12]:.3f} "
        f"({'OK' if ramp[0] == 1.0 and abs(ramp[12] - gain) < 1e-12 else 'FAILED'})"
    )

    # The shape, in the rollout. This is the claim the ramp family rests on and it is not implied by
    # the one above: the gain enters an autoregressive loop, so a narrow early schedule could still
    # have widened the late ensemble through the feedback, or failed to.
    with gain_schedule(model, build_schedule("ramp", gain)), dropout_active(model, DROPOUT):
        ramped = run_arm(noise_std=1.0, injection="independent", seed=0, **shared)

    def ends(traj: np.ndarray) -> tuple[float, float]:
        step_sd = traj[:, :, bool_mask].std(axis=0, ddof=1).mean(axis=1) / target_std
        return float(step_sd[:12].mean()), float(step_sd[12:].mean())

    ramp_early, ramp_late = ends(ramped)
    const_early, const_late = ends(direct)
    ok = ramp_early < const_early and ramp_late > 0.8 * const_late
    notes.append(
        f"the ramp buys the settled year without paying at short lead: spread m1-12 "
        f"{ramp_early:.3f} vs const {const_early:.3f}, m13-24 {ramp_late:.3f} vs const "
        f"{const_late:.3f} ({'OK' if ok else 'FAILED'})"
    )

    drift = max(float((p - before[n]).abs().max()) for n, p in named)
    notes.append(
        f"weights restored bit-exactly after every schedule exits: max |dw| {drift:.2e} "
        f"({'OK' if drift == 0.0 else 'FAILED'})"
    )
    return notes


def score_window(members: np.ndarray, truth: np.ndarray, target_std: float,
                 months: list[tuple[int, int]], climatology: pd.Series) -> dict:
    """Self-anchor this ensemble and score it on every lead block.

    The anchor is re-estimated here for the same reason it is in the sweep: the gain moves the
    settled mean, and reusing another schedule's offset would let that movement be scored as width.
    """
    pred = members.mean(axis=0).mean(axis=1)
    shift = anchor_shift(settled_offset(pred, months, climatology), months, "ramp", seasonal=True)
    shifted = members + shift[:, None]
    out = {}
    for label, block in BLOCKS:
        stats = score(shifted[:, block], truth[block], target_std)
        stats.update(rmse_terms(shifted[:, block], truth[block]))
        stats["spread_sd"] = spread_sd(shifted[:, block], target_std)
        out[label] = stats
    out["offset_y2_sd"] = float(shift[SETTLED_FROM:].mean()) / target_std
    return out


def sweep_window(cfg: DictConfig, year: int, climatology: pd.Series) -> dict:
    """Every family x gain on one window. Returns scores keyed (family, gain), plus the window's
    climate anomaly, which is what the transfer argument turns on."""
    cfg = copy.deepcopy(cfg)
    cfg["dataset"]["starting_year"] = year
    cfg["dataset"]["starting_month"] = 1
    prepared = prepare(cfg)
    if prepared["num_steps"] < STEPS:
        raise ValueError(f"window {year}-01 has {prepared['num_steps']} steps, {STEPS} requested")

    model = prepared["model"]
    shared = dict(prepared["rollout"])
    shared["feature_maps"] = shared["feature_maps"][:STEPS]
    shared["timestamps"] = shared["timestamps"][:STEPS]
    bool_mask, target_std = prepared["bool_mask"], prepared["target_std"]
    truth = prepared["y_true"][:STEPS][:, bool_mask]
    months = window_months(year, 1, STEPS)

    out: dict = {"year": year, "target_std": target_std, "months": months,
                 "num_members": prepared["num_members"], "model": model, "shared": shared,
                 "bool_mask": bool_mask, "truth": truth}
    for family in FAMILIES:
        for gain in GAINS:
            with gain_schedule(model, build_schedule(family, gain)), dropout_active(model, DROPOUT):
                traj = run_arm(noise_std=1.0, injection="independent", seed=0, **shared)
            members = traj[:, :, bool_mask]
            out[(family, gain)] = score_window(members, truth, target_std, months, climatology)
    logger.info(f"window {year}-01 swept: {len(FAMILIES) * len(GAINS)} rollouts")
    return out


def release(window: dict) -> None:
    """Drop a scored window's generator and rollout inputs.

    ``prepare`` loads a fresh checkpoint per window, so holding all 15 would keep 15 copies of the
    generator on the card at once alongside their feature maps. The scores are already extracted by
    the time this is called; only the test window's model is needed afterwards, by
    ``check_schedule``.
    """
    for key in ("model", "shared"):
        window.pop(key, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def argmin_gain(curves: list[dict], family: str, block: str = SELECT_BLOCK) -> float:
    """The CRPS-minimising gain over one or more windows, averaging CRPS across them first.

    Averaging the SCORE and then minimising -- rather than minimising per window and averaging the
    arg -- is what makes a pool behave like one larger sample instead of a vote.
    """
    mean_crps = np.mean([[w[(family, g)][block]["crps"] for g in GAINS] for w in curves], axis=0)
    return float(GAINS[int(np.argmin(mean_crps))])


def clim_gain(curves: list[dict], target: float, family: str = "ramp") -> float:
    """The gain whose settled spread reaches ``target`` sd, averaged over the given windows.

    Uses spread only, so no observation of ``y`` enters. Reported and used as a rule in its own
    right because it is the only candidate available at forecast issue time in the same sense the
    anchor's offset is.
    """
    spread = np.mean([[w[(family, g)]["m13-24"]["spread_sd"] for g in GAINS] for w in curves],
                     axis=0)
    return crossing(np.array(GAINS), spread, target)


def nearest_gain(value: float) -> float:
    """Snap a continuous gain to the swept grid, so every rule is scored on measured rollouts."""
    if not np.isfinite(value):
        return float("nan")
    return float(GAINS[int(np.argmin(np.abs(np.array(GAINS) - value)))])


def report_windows(curves: dict[int, dict], anom: dict[int, float]) -> None:
    """Does the chosen gain hold still across windows, or does it track the window's weather?"""
    print("\n  DOES THE CHOSEN GAIN HOLD STILL?  CRPS-argmin per calibration window, both families")
    print(f"  {'window':16s} {'y2 clim anom':>13s} {'g* const':>9s} {'g* ramp':>9s} "
          f"{'CRPS at g*':>11s} {'CRPS at g=1':>12s} {'split':>7s}")
    print("  " + "-" * 88)
    for year in sorted(curves):
        w = curves[year]
        gc, gr = argmin_gain([w], "const"), argmin_gain([w], "ramp")
        best = w[("ramp", gr)][SELECT_BLOCK]["crps"]
        base = w[("ramp", REFERENCE_GAIN)][SELECT_BLOCK]["crps"]
        tag = "train" if year < POST_TRAIN_FROM else "post"
        print(f"  {year}-01..{year + 1}-12 {anom[year]:+13.3f} {gc:9.2f} {gr:9.2f} "
              f"{best:11.5f} {base:12.5f} {tag:>7s}")
    print("  " + "-" * 88)
    years = sorted(curves)
    gr = np.array([argmin_gain([curves[y]], "ramp") for y in years])
    a = np.array([abs(anom[y]) for y in years])
    if len(years) > 2 and gr.std() > 0 and a.std() > 0:
        r = float(np.corrcoef(a, gr)[0, 1])
        print(f"  correlation between |climate anomaly| and the gain that window asks for: "
              f"r = {r:+.3f}")
        print("  A window with no weather in it needs no spread to cover the weather, so it asks")
        print("  for a small gain. That is the same mechanism that made b_mean untransferable, and")
        print("  it is why no single window can set this parameter.")


def main_report(rows: list[tuple], base: dict, oracle: dict) -> None:
    """The transfer table: what each rule picked, and what it bought on the window it never saw."""
    print(f"\n  SCORED ON TEST  {WINDOWS['test'][0]}  (every gain chosen WITHOUT this window)")
    print(f"  {'rule':26s} {'family':>7s} {'g*':>6s} {'CRPS':>9s} {'vs g=1':>8s} {'RMSE ens':>9s} "
          f"{'sprd/skill':>11s} {'outside':>8s} {'of oracle':>10s}")
    print("  " + "-" * 106)
    span = base["m1-24"]["crps"] - oracle["m1-24"]["crps"]
    for label, family, gain, s in rows:
        got = base["m1-24"]["crps"] - s["m1-24"]["crps"]
        share = got / span if span > 0 else float("nan")
        print(f"  {label:26s} {family:>7s} {gain:6.2f} {s['m1-24']['crps']:9.5f} "
              f"{(s['m1-24']['crps'] / base['m1-24']['crps'] - 1) * 100:+7.1f}% "
              f"{s['m1-24']['rmse_ens']:9.4f} {s['m1-24']['spread_skill']:11.3f} "
              f"{s['m1-24']['extreme_frac'] * 100:7.1f}% {share:9.0%}")
    print("  " + "-" * 106)
    print("  'of oracle' is the share of the achievable CRPS gain this rule actually captured,")
    print("  where the oracle is the gain fitted ON test and is a ceiling, not a candidate. 100%")
    print("  means the rule found the best gain without ever seeing the window it was scored on.")


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Select the gain on calibration windows only, then score the transfer to test."""
    climatology, by_month = observed_climatology(cfg)

    years = [*CALIB_YEARS, TEST_YEAR]
    curves: dict[int, dict] = {}
    for year in years:
        curves[year] = sweep_window(cfg, year, climatology)
        if year != TEST_YEAR:
            release(curves[year])
    test = curves[TEST_YEAR]
    target_std = test["target_std"]
    calib = {y: curves[y] for y in CALIB_YEARS}

    anom = {}
    for y in years:
        months = window_months(y, 1, STEPS)[SETTLED_FROM:]
        anom[y] = float(np.mean([by_month[(yy, mm)] - climatology[mm]
                                 for yy, mm in months])) / target_std

    targets = anomaly_targets(by_month, climatology, target_std)
    notes = check_schedule(test["model"], test["shared"], test["bool_mask"], target_std)

    post = [calib[y] for y in CALIB_YEARS if y >= POST_TRAIN_FROM]
    every = [calib[y] for y in CALIB_YEARS]
    val_w, oper_w = calib[2020], calib[2021]

    rules: list[tuple[str, str, float]] = []
    for family in FAMILIES:
        rules.append(("val  2020-21 only", family, argmin_gain([val_w], family)))
        rules.append(("oper 2021-22 only", family, argmin_gain([oper_w], family)))
        rules.append((f"pool post-train, n={len(post)}", family, argmin_gain(post, family)))
        rules.append((f"pool all, n={len(every)}", family, argmin_gain(every, family)))
    rules.append(("clim 0.307 sd, no y", "ramp",
                  nearest_gain(clim_gain(every, targets["january"]))))
    rules.append(("ORACLE fitted on test", "const", argmin_gain([test], "const")))
    rules.append(("ORACLE fitted on test", "ramp", argmin_gain([test], "ramp")))

    base = test[("const", REFERENCE_GAIN)]
    oracle = test[("ramp", argmin_gain([test], "ramp"))]
    rows = [(label, family, gain, test[(family, gain)])
            for label, family, gain in rules if np.isfinite(gain)]

    print("\n" + "=" * 110)
    print("CHOOSING THE INJECTION GAIN WITHOUT THE TEST WINDOW".center(110))
    print(f"{len(CALIB_YEARS)} January-start calibration windows, {CALIB_YEARS[0]}-"
          f"{CALIB_YEARS[-1]} · selected on {SELECT_BLOCK} CRPS · scored on "
          f"{WINDOWS['test'][0]}".center(110))
    print("=" * 110)
    for note in notes:
        print(f"  check  {note}")

    report_windows(calib, anom)
    main_report(rows, base, oracle)

    print("\n  WHAT EACH RULE COSTS BY LEAD BLOCK  (CRPS, and the change against g = 1)")
    print(f"  {'rule':26s} {'family':>7s} {'g*':>6s}" +
          "".join(f"{b:>20s}" for b, _s in BLOCKS))
    print("  " + "-" * 101)
    for label, family, gain, s in rows:
        cells = "".join(
            f"{s[b]['crps']:11.5f}{(s[b]['crps'] / base[b]['crps'] - 1) * 100:+8.1f}%"
            for b, _sl in BLOCKS)
        print(f"  {label:26s} {family:>7s} {gain:6.2f}" + cells)
    print("  " + "-" * 101)
    print("  The const family is dominated by the short lead, where the ensemble is supposed to be")
    print("  narrow; the ramp buys the settled year without paying for it at m1-12. If a rule's")
    print("  m1-24 gain comes entirely from m13-24, that is the ramp doing what it was built for.")

    print(f"\n  THE WINDOWS THEMSELVES  (year-2 climate anomaly, in sd of targets_std "
          f"{target_std:.4f})")
    print(f"  calibration {CALIB_YEARS[0]}-{CALIB_YEARS[-1]}: mean "
          f"{np.mean([anom[y] for y in CALIB_YEARS]):+.3f} sd, "
          f"sd {np.std([anom[y] for y in CALIB_YEARS], ddof=1):.3f} sd")
    print(f"  val {anom[2020]:+.3f} · oper {anom[2021]:+.3f} · TEST {anom[TEST_YEAR]:+.3f}")
    print(f"  climatological sd of that quantity, 65 January-start blocks: "
          f"{targets['january']:.3f} sd")
    print("=" * 110 + "\n")

    out = cfg["model"].get("saving_path")
    if not out:
        return
    saved: dict[str, np.ndarray] = {
        "gains": np.array(GAINS),
        "calib_years": np.array(CALIB_YEARS),
        "families": np.array(FAMILIES),
        "rule_labels": np.array([r[0] for r in rows]),
        "rule_family": np.array([r[1] for r in rows]),
        "rule_gain": np.array([r[2] for r in rows]),
        "anom": np.array([anom[y] for y in years]),
        "anom_years": np.array(years),
        "target_std": np.array(target_std),
        "noise_weights_rms": np.array(weights_rms(test["model"])),
        "target_january": np.array(targets["january"]),
    }
    for family in FAMILIES:
        for block, _sl in BLOCKS:
            saved[f"calib_{family}_{block}_crps"] = np.array(
                [[calib[y][(family, g)][block]["crps"] for g in GAINS] for y in CALIB_YEARS])
            saved[f"calib_{family}_{block}_spread"] = np.array(
                [[calib[y][(family, g)][block]["spread_sd"] for g in GAINS] for y in CALIB_YEARS])
            saved[f"test_{family}_{block}_crps"] = np.array(
                [test[(family, g)][block]["crps"] for g in GAINS])
            saved[f"test_{family}_{block}_spread"] = np.array(
                [test[(family, g)][block]["spread_sd"] for g in GAINS])
            saved[f"test_{family}_{block}_rmse_ens"] = np.array(
                [test[(family, g)][block]["rmse_ens"] for g in GAINS])
            saved[f"test_{family}_{block}_spread_skill"] = np.array(
                [test[(family, g)][block]["spread_skill"] for g in GAINS])
            saved[f"test_{family}_{block}_extreme_frac"] = np.array(
                [test[(family, g)][block]["extreme_frac"] for g in GAINS])
    for block, _sl in BLOCKS:
        saved[f"rule_{block}_crps"] = np.array([r[3][block]["crps"] for r in rows])
        saved[f"rule_{block}_rmse_ens"] = np.array([r[3][block]["rmse_ens"] for r in rows])
    np.savez(f"{out}_transfer.npz", **saved)
    logger.info(f"Saved to '{out}_transfer.npz'.")


if __name__ == "__main__":
    main()
