"""Proper scoring for the ensemble: CRPS, rank histogram, spread-skill ratio.

WHY THIS EXISTS. Rounds 9-11 were all scored on envelope coverage -- the share of pixel-months
falling inside the ensemble's min-max range. That was defensible while every ensemble was
severely under-dispersed (24.8% to 31.6% against the ~90% a calibrated 20-member envelope would
give), because in that regime more spread is unambiguously better. Round 11 took coverage to
74.5%, and at that point the assumption expires: coverage rises monotonically with width, so it
cannot distinguish a better-calibrated ensemble from a merely wider one, and it cannot see
over-dispersion at all.

CRPS is a PROPER scoring rule. It is minimised by the true predictive distribution, so it
penalises an ensemble that is too wide exactly as it penalises one that is too narrow, and it
trades that off against accuracy on a single scale. Lower is better.

WHAT IS COMPUTED.

  crps          Fair (unbiased) ensemble CRPS. For members x_1..x_M and observation y:
                    CRPS = (1/M) sum_m |x_m - y|  -  (1/(M(M-1))) sum_{i<j} |x_i - x_j|
                The 1/(M(M-1)) normalisation is the fair estimator of Ferro et al. [2008]; the
                biased 1/(2M^2) form rewards under-dispersion at small M, which is precisely the
                failure mode under investigation, so it would beg the question. M = 20 here.
  crps_det      CRPS of a deterministic forecast at the ensemble mean, which for a point forecast
                reduces to MAE. This is the competitor the spread has to beat.
  crps_skill    1 - crps/crps_det. POSITIVE means the ensemble's spread earns its keep; negative
                means you would be better off shipping the ensemble mean as a point forecast.
  rank_hist     Talagrand diagram: for each observation, how many members fall below it (0..M),
                with ties broken at random under a fixed seed. Flat = calibrated. U-shaped =
                under-dispersed. Dome-shaped = over-dispersed. Sloped = biased.
  reliability   sum_k |p_k - 1/(M+1)| over the M+1 bins. 0 is a perfectly flat histogram; the
                worst case (all mass in one bin) is 2(1 - 1/(M+1)) = 1.905 at M = 20.
  extreme_frac  p_0 + p_M, the share of observations falling outside the envelope entirely.
                Calibrated value is 2/(M+1) = 9.5%. This is 1 - coverage, and is reported so the
                old coverage numbers stay readable against the new ones.
  spread_skill  sqrt( ((M+1)/M) * mean ensemble variance / MSE of the ensemble mean ).
                1.0 is calibrated, < 1 under-dispersed, > 1 over-dispersed. The (M+1)/M factor is
                the finite-ensemble correction.

CRPS and crps_det are reported in standardised units (divided by targets_std) so they sit on the
same scale as the spreads quoted throughout rounds 9-11.
"""

import logging
from contextlib import contextmanager

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from utils.dropout_probe import dropout_active
from utils.stochasticity_probe import prepare, run_arm

logger = logging.getLogger(__file__)

RANK_SEED = 0


def crps_fair(members: np.ndarray, truth: np.ndarray) -> float:
    """Fair ensemble CRPS, averaged over all points. ``members`` (M, ...), ``truth`` (...)."""
    m = members.shape[0]
    if m < 2:
        raise ValueError("fair CRPS needs at least 2 members")
    skill = np.abs(members - truth[None]).mean(axis=0)

    # sum_{i<j} |x_i - x_j| via the sorted-order identity, which avoids the M^2 pairwise array:
    #   sum_{i<j} (x_(j) - x_(i)) = sum_k (2k - (M-1)) * x_(k)   for ascending x_(k)
    ordered = np.sort(members, axis=0)
    weights = (2 * np.arange(m) - (m - 1)).reshape(-1, *([1] * (members.ndim - 1)))
    pair_sum = (weights * ordered).sum(axis=0)
    spread_term = pair_sum / (m * (m - 1))
    return float((skill - spread_term).mean())


def rank_histogram(members: np.ndarray, truth: np.ndarray, seed: int = RANK_SEED) -> np.ndarray:
    """Counts per rank bin, 0..M. Ties broken at random so the histogram is not biased by them."""
    m = members.shape[0]
    rng = np.random.default_rng(seed)
    below = (members < truth[None]).sum(axis=0)
    ties = (members == truth[None]).sum(axis=0)
    # distribute each tie uniformly over the ranks it is consistent with
    extra = rng.integers(0, ties + 1) if np.any(ties) else np.zeros_like(below)
    ranks = (below + extra).ravel()
    return np.bincount(ranks, minlength=m + 1)[: m + 1]


def score(members: np.ndarray, truth: np.ndarray, target_std: float) -> dict:
    """Every score for one ensemble. ``members`` (M, T, P) already masked; ``truth`` (T, P)."""
    m = members.shape[0]
    crps = crps_fair(members, truth) / target_std
    ens_mean = members.mean(axis=0)
    crps_det = float(np.abs(ens_mean - truth).mean()) / target_std

    counts = rank_histogram(members, truth)
    p = counts / counts.sum()
    reliability = float(np.abs(p - 1.0 / (m + 1)).sum())
    extreme = float(p[0] + p[-1])

    mse_mean = float(((ens_mean - truth) ** 2).mean())
    mean_var = float(members.var(axis=0, ddof=1).mean())
    spread_skill = float(np.sqrt(((m + 1) / m) * mean_var / mse_mean)) if mse_mean > 0 else 0.0

    return {
        "crps": crps,
        "crps_det": crps_det,
        "crps_skill": 1.0 - crps / crps_det if crps_det > 0 else float("nan"),
        "rank_hist": p,
        "reliability": reliability,
        "extreme_frac": extreme,
        "coverage": 1.0 - extreme,
        "spread_skill": spread_skill,
    }


@contextmanager
def _noop():
    yield


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Score the shipping ensemble with inference dropout off and on."""
    prepared = prepare(cfg)
    shared = prepared["rollout"]
    model = prepared["model"]
    y_true, bool_mask = prepared["y_true"], prepared["bool_mask"]
    target_std = prepared["target_std"]
    truth = y_true[:, bool_mask]

    rows = []
    for label, drop in [("ships today   dropout off", False), ("ships +       dropout ON ", True)]:
        ctx = dropout_active(model, True) if drop else _noop()
        with ctx:
            traj = run_arm(noise_std=1.0, injection="independent", seed=0, **shared)
        members = traj[:, :, bool_mask]
        rows.append((label, score(members, truth, target_std)))

    m = prepared["num_members"]
    print("\n" + "=" * 100)
    print("PROPER SCORES".center(100))
    print(f"{m} members · {prepared['num_steps']} rollout steps · CRPS in standardised units".center(100))
    print("=" * 100)
    print(f"{'arm':26s} {'CRPS':>9s} {'CRPS det':>9s} {'skill':>8s} {'spread/skill':>13s} "
          f"{'reliab.':>9s} {'outside':>9s}")
    print("-" * 100)
    for label, s in rows:
        print(f"{label:26s} {s['crps']:9.5f} {s['crps_det']:9.5f} {s['crps_skill']:+8.3f} "
              f"{s['spread_skill']:13.3f} {s['reliability']:9.3f} {s['extreme_frac'] * 100:8.1f}%")
    print("-" * 100)
    print("skill        1 - CRPS/CRPS_det. Positive = the spread beats shipping the ensemble mean.")
    print("spread/skill 1.0 = calibrated, <1 under-dispersed, >1 over-dispersed.")
    print("reliab.      flatness of the rank histogram; 0 is perfect, 1.905 is the worst case.")
    print("outside      share of observations outside the envelope; calibrated is 9.5%.")

    for label, s in rows:
        h = s["rank_hist"]
        bars = "".join("▁▂▃▄▅▆▇█"[min(7, int(v / max(h.max(), 1e-9) * 7.999))] for v in h)
        print(f"\n{label}  rank histogram (bin 0 = obs below all members, bin {m} = above all)")
        print(f"  {bars}   flat would be {1 / (m + 1) * 100:.1f}% per bin; "
              f"ends {h[0] * 100:.1f}% / {h[-1] * 100:.1f}%")
    print("=" * 100)

    out = cfg["model"].get("saving_path")
    if out:
        np.savez(
            f"{out}_scores.npz",
            arms=np.array([r[0] for r in rows]),
            **{
                k: np.array([r[1][k] for r in rows])
                for k in ("crps", "crps_det", "crps_skill", "reliability",
                          "extreme_frac", "coverage", "spread_skill")
            },
            rank_hist=np.stack([r[1]["rank_hist"] for r in rows]),
        )
        logger.info(f"Saved to '{out}_scores.npz'.")


if __name__ == "__main__":
    main()