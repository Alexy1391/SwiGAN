"""Global bias correction estimated on validation, scored on the test rollout.

WHY THIS EXISTS. ``utils/ensemble_scores.py`` showed the rank histograms are not U-shaped, they
are right-skewed: at the round-8 seed-60 baseline the observation lies ABOVE all 20 members 64.2%
of the time, against the 4.76% a calibrated 20-member ensemble puts in each tail. That is a
LOCATION error, not a WIDTH error. Envelope coverage cannot tell the two apart -- a biased
ensemble has low coverage, and widening it does raise coverage -- which is how rounds 9-11 could
read as progress while partly inflating variance to paper over an offset.

Subtracting a single scalar from the seed-60 rollout improved CRPS 23.1%, but that scalar was
estimated on the very rollout it was scored against. It is an upper bound, not a result. This
script estimates the offset where it can honestly be estimated -- on the validation split -- and
then scores the transfer to the test window. The gap between the two is the whole point: it is
what separates "the model has a fixable bias" from "a free parameter fitted to the test set".

WINDOWS. Reconstructed from the split the training code actually builds: ``build_train_val_test_
datasets`` cuts the 780 months of the parquet chronologically at ``train_ratio`` 0.92 and
``val_ratio`` 0.04, giving train 1960-01..2019-09, val 2019-10..2022-04, test 2022-05..2024-12.

  val    2020-01 .. 2021-12   24 steps, entirely inside the validation split
  oper   2021-01 .. 2022-12   24 steps, the last 24 months before the forecast starts
  test   2023-01 .. 2024-12   24 steps, the window every round has been scored on

All three start in January and all three run 24 months. That is deliberate. It means lead time k
lands on the same calendar month in every window, and 24 months is two whole annual cycles, so the
seasonal cycle cancels out of any mean taken over a window. Neither property is needed for a
single scalar, but both are what make the PER-LEAD-TIME offset that comes next well posed: on
mismatched start months, lead time and season are confounded and a per-lead-time table would be
fitting season under another name. The per-lead-time and per-pixel residuals are saved here so
that step is a read, not another rollout.

TWO CALIBRATION PROTOCOLS, because they answer different questions. ``val`` is the split the
model was validated on, which is what "estimate the offset on validation" means literally. ``oper``
is the 24 months of observations that would actually be in hand at the moment the 2023-01 forecast
is issued. It straddles the val/test cut at 2022-05, but it contains no information whatever from
the scored window, so it does not leak -- and if the bias drifts with distance from the training
data, it is the protocol a deployed system would use and the only one that can keep up.

The scored checkpoint is ``last.ckpt``, which ``save_last=True`` writes by epoch count. The
``monitor="val/generator_rmse_epoch"`` selection only governs the top-k files, which are not what
is scored -- so the validation split did not choose these weights, and an offset estimated on it
is not fitted to them.

ESTIMATORS. Two constant offsets, both ADDED to every member:

  b_mean   mean(y - ensemble_mean) over the calibration window. The offset that zeroes the mean
           error, hence the MSE-optimal constant shift for the ensemble mean. This is "the bias".
  b_crps   median over all (member, step, pixel) residuals y - x_m. This is EXACTLY the
           CRPS-optimal constant shift, not an approximation to it: the fair CRPS is
           mean_m |x_m - y| - (1/(M(M-1))) sum_{i<j} |x_i - x_j|, and the second term is invariant
           under a shift common to every member. So only the mean-absolute-deviation term moves,
           and that is minimised at the median. Reported because b_mean is the natural definition
           of a bias while b_crps is the one the score actually wants, and a right-skewed error
           distribution is exactly the case where they disagree.

  b_oracle, the same b_mean re-estimated on the test window itself, is scored alongside as the
  ceiling. It is not a candidate. It is there so the transfer loss is visible rather than assumed.

NO CLIPPING. swi_u is not bounded at 1 in this dataset: it runs [-0.027, 1.273] and 13.0% of
observations exceed 1.0. There is therefore no physical bound to clip a shifted ensemble against,
and clipping is not applied. This also keeps b_crps exactly optimal rather than approximately so,
since a clip would break the shift-invariance the derivation above rests on.
"""

import copy
import logging

import hydra
import numpy as np
from omegaconf import DictConfig

from utils.dropout_probe import dropout_active
from utils.ensemble_scores import crps_fair, score
from utils.stochasticity_probe import prepare, run_arm

logger = logging.getLogger(__file__)

# key -> (label, starting_year, starting_month, num_steps)
WINDOWS = {
    "val": ("2020-01..2021-12 val", 2020, 1, 24),
    "oper": ("2021-01..2022-12 oper", 2021, 1, 24),
    "test": ("2023-01..2024-12 test", 2023, 1, 24),
}


def rollout(cfg: DictConfig, year: int, month: int, steps: int, seed: int = 0) -> dict:
    """Both dropout arms over one window. Returns members (M, T, P) per arm, truth, and the scale.

    ``prepare`` rolls out to the end of the parquet from whatever month it is pointed at, so the
    inputs are truncated to ``steps`` here rather than by config. Everything else -- warm start,
    masking, feature standardisation, eval mode -- is left exactly as the shipping inference path
    builds it, so these numbers stay comparable with rounds 8-11. Both arms share one ``prepare``
    because loading the parquet dominates the cost of a 24-step rollout.
    """
    cfg = copy.deepcopy(cfg)
    cfg["dataset"]["starting_year"] = year
    cfg["dataset"]["starting_month"] = month
    prepared = prepare(cfg)

    shared = dict(prepared["rollout"])
    available = prepared["num_steps"]
    if available < steps:
        raise ValueError(f"window {year}-{month:02d} has {available} steps, {steps} requested")
    shared["feature_maps"] = shared["feature_maps"][:steps]
    shared["timestamps"] = shared["timestamps"][:steps]

    bool_mask = prepared["bool_mask"]
    out = {"truth": prepared["y_true"][:steps][:, bool_mask], "target_std": prepared["target_std"]}
    for dropout in (False, True):
        ctx = dropout_active(prepared["model"], True) if dropout else None
        if ctx is not None:
            ctx.__enter__()
        try:
            traj = run_arm(noise_std=1.0, injection="independent", seed=seed, **shared)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        out["on" if dropout else "off"] = traj[:, :, bool_mask]
    return out


def offsets(members: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """The two constant offsets, in raw swi_u units. Both are ADDED to every member."""
    residual_mean = truth - members.mean(axis=0)  # (T, P)
    residual_all = truth[None] - members  # (M, T, P)
    return {
        "b_mean": float(residual_mean.mean()),
        "b_crps": float(np.median(residual_all)),
    }


def rmse_terms(members: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Accuracy readouts. ``members`` (M, T, P), ``truth`` (T, P).

    ``rmse_ens`` is the RMSE of the ensemble mean -- the deterministic accuracy a bias correction
    acts on directly. ``rmse_mem`` and ``rmse_max`` are the member-averaged PER-PIXEL RMSE, pooled
    and worst-pixel, which is the quantity rounds 9-11 quoted as ``rmse mean`` / ``rmse max``; it
    averages errors across members, never predictions, so it does not benefit from ensembling.
    """
    ens_mean = members.mean(axis=0)
    per_pixel = np.sqrt(((members - truth[None]) ** 2).mean(axis=1)).mean(axis=0)  # (P,)
    return {
        "rmse_ens": float(np.sqrt(((ens_mean - truth) ** 2).mean())),
        "rmse_mem": float(per_pixel.mean()),
        "rmse_max": float(per_pixel.max()),
    }


def score_shifted(members: np.ndarray, truth: np.ndarray, target_std: float, b: float) -> dict:
    """Every score for the ensemble shifted by a constant ``b``, plus its residual bias."""
    shifted = members + b
    out = score(shifted, truth, target_std)
    out.update(rmse_terms(shifted, truth))
    out["bias_sd"] = float((truth - shifted.mean(axis=0)).mean()) / target_std
    out["offset"] = b
    return out


def check_estimators(members: np.ndarray, truth: np.ndarray, b: dict[str, float]) -> list[str]:
    """Verify the two offsets do on the calibration set what their derivations claim.

    b_mean must zero the mean error exactly. b_crps must sit at the minimum of CRPS(b), which is
    checked against a direct scan rather than assumed -- the closed form relies on the fair CRPS
    spread term being shift-invariant, and that is worth confirming numerically once.
    """
    notes = []
    residual = float((truth - (members + b["b_mean"]).mean(axis=0)).mean())
    notes.append(
        f"b_mean zeroes the calibration mean error: residual {residual:+.2e} "
        f"({'OK' if abs(residual) < 1e-6 else 'FAILED'})"
    )

    grid = b["b_crps"] + np.linspace(-0.05, 0.05, 21)
    curve = [crps_fair(members + g, truth) for g in grid]
    best = float(grid[int(np.argmin(curve))])
    notes.append(
        f"b_crps sits at the CRPS minimum: closed form {b['b_crps']:+.5f}, scan argmin "
        f"{best:+.5f}, gap {abs(best - b['b_crps']):.2e} "
        f"({'OK' if abs(best - b['b_crps']) <= 0.0051 else 'FAILED'})"
    )
    return notes


ROW = ("{:<28s} {:>8.5f} {:>9.5f} {:>+8.3f} {:>11.3f} {:>8.3f} {:>7.1f}% "
       "{:>9.4f} {:>9.4f} {:>+8.3f}")
HEAD = ("{:<28s} {:>8s} {:>9s} {:>8s} {:>11s} {:>8s} {:>8s} {:>9s} {:>9s} {:>8s}").format(
    "arm", "CRPS", "CRPS det", "skill", "sprd/skill", "reliab", "outside", "RMSE ens", "RMSE mem",
    "bias sd")


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Estimate the offset on each calibration window, apply it to the test rollout, score."""
    runs = {key: rollout(cfg, year, month, steps)
            for key, (_, year, month, steps) in WINDOWS.items()}

    saved = {"arms": np.array(["uncorrected", "val b_mean", "val b_crps", "oper b_mean",
                               "oper b_crps", "oracle"])}
    for key in ("off", "on"):
        arm = "dropout ON " if key == "on" else "dropout off"
        target_std = runs["test"]["target_std"]
        test_members, test_truth = runs["test"][key], runs["test"]["truth"]

        b = {w: offsets(runs[w][key], runs[w]["truth"]) for w in WINDOWS}
        notes = check_estimators(runs["val"][key], runs["val"]["truth"], b["val"])

        def on_test(shift: float, mem: np.ndarray = test_members,
                    tru: np.ndarray = test_truth, std: float = target_std) -> dict:
            return score_shifted(mem, tru, std, shift)

        rows = [
            ("uncorrected", on_test(0.0)),
            ("+ b_mean   val 2020-21", on_test(b["val"]["b_mean"])),
            ("+ b_crps   val 2020-21", on_test(b["val"]["b_crps"])),
            ("+ b_mean   oper 2021-22", on_test(b["oper"]["b_mean"])),
            ("+ b_crps   oper 2021-22", on_test(b["oper"]["b_crps"])),
            ("+ b_oracle test, ceiling", on_test(b["test"]["b_mean"])),
        ]

        print("\n" + "=" * 122)
        print(f"GLOBAL BIAS CORRECTION  ·  {arm}".center(122))
        print("offset estimated on a calibration window, scored on 2023-01..2024-12".center(122))
        print("=" * 122)
        for note in notes:
            print(f"  check  {note}")

        print(f"\n  THE BIAS ITSELF, per window (mean of y - ensemble mean, in sd of "
              f"targets_std {target_std:.4f})")
        print(f"  {'window':26s} {'b_mean':>10s} {'b_mean sd':>11s} {'b_crps sd':>11s} "
              f"{'lead 1-6':>10s} {'lead 19-24':>11s}")
        print("  " + "-" * 84)
        for w, (label, *_rest) in WINDOWS.items():
            resid = runs[w]["truth"] - runs[w][key].mean(axis=0)
            by_lead = resid.mean(axis=1) / target_std
            print(f"  {label:26s} {b[w]['b_mean']:+10.5f} {b[w]['b_mean'] / target_std:+11.3f} "
                  f"{b[w]['b_crps'] / target_std:+11.3f} {by_lead[:6].mean():+10.3f} "
                  f"{by_lead[18:].mean():+11.3f}")
        print("  " + "-" * 84)
        gap_val = (b["test"]["b_mean"] - b["val"]["b_mean"]) / target_std
        gap_oper = (b["test"]["b_mean"] - b["oper"]["b_mean"]) / target_std
        print(f"  transfer gap  val -> test {gap_val:+.3f} sd    oper -> test {gap_oper:+.3f} sd")

        print(f"\n  SCORED ON THE TEST WINDOW  {WINDOWS['test'][0]}")
        print("  " + HEAD)
        print("  " + "-" * 120)
        base = rows[0][1]
        for label, s in rows:
            print("  " + ROW.format(label, s["crps"], s["crps_det"], s["crps_skill"],
                                    s["spread_skill"], s["reliability"], s["extreme_frac"] * 100,
                                    s["rmse_ens"], s["rmse_mem"], s["bias_sd"]))
        print("  " + "-" * 120)
        for label, s in rows[1:]:
            hist, base_hist = s["rank_hist"], base["rank_hist"]
            print(f"  {label:<26s} CRPS {(s['crps'] / base['crps'] - 1) * 100:+6.1f}%   "
                  f"RMSE ens {(s['rmse_ens'] / base['rmse_ens'] - 1) * 100:+6.1f}%   "
                  f"reliab {base['reliability']:.3f}->{s['reliability']:.3f}   "
                  f"bin 0 {base_hist[0] * 100:4.1f}->{hist[0] * 100:4.1f}%   "
                  f"bin 20 {base_hist[-1] * 100:4.1f}->{hist[-1] * 100:4.1f}%")
        print("=" * 122)

        saved[f"{key}_target_std"] = target_std
        for w in WINDOWS:
            saved[f"{key}_{w}_b_mean"] = b[w]["b_mean"]
            saved[f"{key}_{w}_b_crps"] = b[w]["b_crps"]
            # Residual structure, so per-lead-time and per-pixel are a read, not another rollout.
            # Lead time k is the same calendar month in every window by construction.
            resid = runs[w]["truth"] - runs[w][key].mean(axis=0)  # (T, P)
            saved[f"{key}_{w}_bias_by_lead"] = resid.mean(axis=1)
            saved[f"{key}_{w}_bias_by_pixel"] = resid.mean(axis=0)
        for field in ("crps", "crps_det", "crps_skill", "spread_skill", "reliability",
                      "extreme_frac", "coverage", "rmse_ens", "rmse_mem", "rmse_max", "bias_sd",
                      "offset"):
            saved[f"{key}_{field}"] = np.array([s[field] for _, s in rows])
        saved[f"{key}_rank_hist"] = np.stack([s["rank_hist"] for _, s in rows])

    out = cfg["model"].get("saving_path")
    if out:
        np.savez(f"{out}_bias.npz", **saved)
        logger.info(f"Saved to '{out}_bias.npz'.")


if __name__ == "__main__":
    main()