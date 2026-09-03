"""Sweep a global gain on the injected noise, over the climatology-anchored ensemble.

WHY THIS IS ONLY NOW WELL POSED. Rounds 9-11 were criticised for reading envelope coverage, which
rises monotonically with width and therefore cannot tell a better-calibrated ensemble from a merely
wider one. Widening the ensemble while a LOCATION error was still in it would have been that
mistake again in a new costume: the extra width would have been covering the model's drift.
``utils/climatology_anchor.py`` removed the model-attributable half of that error without seeing
``swi_u`` over the scored window, and showed that what remains at long lead is the window's CLIMATE
ANOMALY -- a property of the weather, identical for all seven checkpoints, contained in no window
that ends before the forecast starts. That residual is genuinely unpredictable at issue time, and a
calibrated ensemble is supposed to cover it. So width is now a legitimate thing to be short of.

THE TARGET IS COMPUTED HERE, NOT TUNED. Two routes, both printed below:

  route A  climatological. The anchored long-lead residual IS the 12-month-mean climate anomaly, so
           the ensemble must be about as wide as that anomaly's climatological sd. That sd is a
           property of the observed record and contains nothing from any model, so it is a target
           and not a knob. IT MATTERS HOW IT IS ESTIMATED. Over the three scored windows alone
           (+0.055 / -0.543 / +0.442 sd) it comes out at 0.496 sd -- but n = 3, and those three
           windows are not typical: over all 65 January-start years in the record the same
           quantity has sd 0.307 sd, and that estimate is stable across eras (0.27-0.31 for
           1990+, 2000+, and the training era alone). The three scored windows happen to sit at
           1.4-1.8 sigma of their own distribution. The 65-year figure is the one to aim at; the
           n = 3 figure is reported beside it because it is the number the sweep was commissioned
           against, and the gap between them is a full factor of 1.6 in the required gain.
  route B  spread-skill. The ratio is already computed per checkpoint by ``utils/ensemble_scores``
           and is 1.0 when calibrated, so the deficit is 1/ratio directly.

The two are NOT independent, and they are not even the same quantity. Route B's denominator is the
MSE of the anchored ensemble mean, which at long lead is dominated by the same climate anomaly route
A takes the sd of. But route A's target is the sd of the WINDOW-MEAN anomaly -- one number per
window -- while the spread it is compared against is the across-member sd at each pixel-month, which
must cover the per-pixel and month-to-month departures as well. So ROUTE A IS A FLOOR, not a target:
an ensemble narrower than it cannot be calibrated, an ensemble wider than it may still be too
narrow. Route B is the criterion. Where the two disagree, that is the structure and not a
contradiction, and the disagreement is reported rather than averaged away.

WHY THE GAIN, AND NOT ``noise_std``. The generator has two noise sources and ``noise_std`` scales
only the smaller one. ``utils/stochasticity_probe.py`` measured the split: the bottleneck vector z
carries ~0.2% of the ensemble variance (injection-only 0.13551 against ships-today 0.13567), and
cranking it to std=20 buys 1.35x. The 20 ``noise_weights`` tensors -- 3840 per-channel gains at the
two injection sites of every encoder and decoder block -- carry essentially all of it, and only two
points on that axis have ever been measured: g = 0 (``injection_off``) and g = 1. ``noise_gain``
below is the missing sweep, and it is ten lines because ``injection_off`` is the g = 0 case of it.

THE ANCHOR IS RE-ESTIMATED AT EVERY GAIN, which is what keeps the two knobs identifiable. The
settled model offset is read off the rollout being corrected -- legitimately, since no observation
of the predictand enters it -- so changing the gain changes the rollout and therefore changes the
offset. Reusing the g = 1 offset would let a gain that happened to shift the settled mean be scored
as if it had bought width, and CRPS would reward it: exactly the confound the anchor was built to
remove. Anchor sets the mean, gain sets the width, in that order, at every point of the sweep. The
amount the anchor had to move is printed per gain (``shift y2``) so a mean that drifts under gain is
visible rather than absorbed.

TWO THINGS THAT WILL OTHERWISE FOOL THE READING.

  The noise is load-bearing, not decoration. The deterministic control has rmse_mean 0.4342 against
  0.2564 at g = 1: REMOVING the injection degrades the point forecast by 70%. Scaling g moves 20
  internal sites off their trained operating point, compounded through 24 autoregressive steps, so
  the mean should be expected to degrade before the spread reaches target. Spread alone will look
  like success right up to the point the forecast falls apart, so CRPS -- which is proper, and
  trades width against accuracy on one scale -- and RMSE are scored together at every gain, and the
  g = 0 row is kept in the sweep as the standing reminder of what removing the injection costs.

  One global gain probably cannot serve both ends of the lead time. At m1-12 the initial condition
  is still live and the ensemble should be narrow; at m13-24 it needs to cover the climate anomaly.
  A constant g scales both. Everything is therefore reported lead-resolved, and if the required
  gains split across blocks the fix is a lead-dependent inflation mirroring the ramp already in the
  anchor -- not a compromise value of one constant.

WHAT A NEGATIVE RESULT WOULD MEAN. If the required gain wrecks the mean before the spread arrives
-- the likely outcome for the r8/r9 checkpoints, whose deficit is 3.5-3.8x on route B -- that says
dispersion is not fixable at inference on those weights, and redirects to the training objective (a
proper-scoring term in the loss) rather than to another inference knob. Inference-only either way:
no retraining, ~1 min per checkpoint per gain.
"""

import copy
import logging
from contextlib import contextmanager
from pathlib import Path

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
from utils.stochasticity_probe import prepare, run_arm

logger = logging.getLogger(__file__)

# The swept gains. g=1 is what ships; g=0 is exactly ``injection_off`` and is kept in the sweep as
# the cost-of-removing-the-injection reference. The grid is dense from 1 to 4 because every route's
# target lands in there, and thins out above it where only the shape of the failure matters.
GAINS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
REFERENCE_GAIN = 1.0
REPEAT_SEEDS = (0, 1, 2)  # repeats at the reference gain, to size the sweep's own noise
SCORED_WINDOW = "test"
DROPOUT = True  # inference dropout ON: settled 7 of 7 by utils/ensemble_scores.py

# Lead blocks. The first three are the headline split the sweep is read on; the four sixes resolve
# it further, which is what a lead-dependent inflation would have to be built from.
BLOCKS: tuple[tuple[str, slice], ...] = (
    ("m1-24", slice(0, 24)),
    ("m1-12", slice(0, 12)),
    ("m13-24", slice(12, 24)),
)
FINE_BLOCKS: tuple[tuple[str, slice], ...] = (
    ("m1-6", slice(0, 6)),
    ("m7-12", slice(6, 12)),
    ("m13-18", slice(12, 18)),
    ("m19-24", slice(18, 24)),
)
ALL_BLOCKS = BLOCKS + FINE_BLOCKS
ANOMALY_BLOCK = 12  # months averaged over for the climatological target; year 2 of the window


@contextmanager
def noise_gain(model: torch.nn.Module, gain: float):
    """Multiply every ``noise_weights`` parameter by ``gain``, restoring them exactly on exit.

    The injected field enters as ``out + noise * noise_weights``, so scaling the weights scales the
    injected noise at all 20 sites and nothing else -- the conv path, the attention and the skip
    connections are untouched. ``gain=0`` is bit-identical to ``utils.stochasticity_probe.
    injection_off``, which is checked numerically in ``check_gain`` rather than asserted.

    Restoration is a ``copy_`` from a saved clone rather than a division by ``gain``, so a sweep of
    many gains cannot accumulate floating-point drift in the weights and ``gain=0`` is invertible.
    """
    saved = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if "noise_weights" in name
    }
    if not saved:
        raise RuntimeError("no noise_weights parameters found -- the model is not the SwiGAN UNet")
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "noise_weights" in name:
                param.mul_(gain)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "noise_weights" in name:
                    param.copy_(saved[name])


def weights_rms(model: torch.nn.Module) -> float:
    """Rms of every ``noise_weights`` parameter -- the size of the lever the gain multiplies.

    Reported and saved because it, not the spread-skill ratio, is what decides whether the sweep can
    do anything at all on a given checkpoint. A multiplier on 2e-4 is still 2e-4.

    It is the FINAL magnitude that matters, not the init: ``randn`` init lands at ~1.0 and barely
    moves over a run, but ``zeros`` init lands wherever its learning-rate budget carries it -- 2e-4
    at lr scale 1 (round 9 arm A), 0.018 at x100 (arm B), and 1.47 at x10000 (round 10 arm C), which
    is the largest of the seven checkpoints. Reading the init instead of the weights would put arm C
    in the wrong column.
    """
    return float(
        torch.cat([
            p.detach().flatten() for n, p in model.named_parameters() if "noise_weights" in n
        ]).pow(2).mean().sqrt()
    )


def check_gain(model: torch.nn.Module) -> list[str]:
    """Verify the context manager does what it claims, before any of its output is interpreted."""
    notes = []
    named = [(n, p) for n, p in model.named_parameters() if "noise_weights" in n]
    channels = sum(int(p.numel()) for _n, p in named)
    notes.append(f"the gain reaches {len(named)} noise_weights tensors, {channels} per-channel "
                 f"gains")

    # The size of the lever, before anything is swept. A gain is a MULTIPLIER, so on weights that
    # ended near 0 it is structurally inert: g x 0 is 0 at every g, and no sweep can find width the
    # weights do not carry. Round 9 arm A sits at rms 2e-4, so a flat sweep there is the lever being
    # absent rather than the hypothesis failing.
    #
    # This reads the FINAL magnitude and says nothing about the init, because the init does not
    # predict it. Round 10 arm C is zeros-init and ends at rms 1.47 -- the LARGEST of the seven --
    # because its lr scale of 10000 let the gains actually learn a scale, which was that arm's
    # point. "zeros-init" and "inert" are different claims and must not be printed as one.
    rms = weights_rms(model)
    verdict = ("the gain has something to scale" if rms > 0.1
               else "NEAR ZERO: these gains ended at ~0, so the gain multiplier is inert here")
    notes.append(f"rms(noise_weights) {rms:.6f} -- {verdict}")

    before = {n: p.detach().clone() for n, p in named}
    with noise_gain(model, 0.0):
        worst = max(float(p.abs().max()) for _n, p in named)
    notes.append(
        f"g=0 zeroes them exactly, reproducing injection_off: max |w| {worst:.2e} "
        f"({'OK' if worst == 0.0 else 'FAILED'})"
    )

    with noise_gain(model, 3.7):
        worst_scale = max(
            float((p - before[n] * 3.7).abs().max()) for n, p in named
        )
    notes.append(
        f"g=3.7 scales them by exactly 3.7: max |w - 3.7 w0| {worst_scale:.2e} "
        f"({'OK' if worst_scale < 1e-6 else 'FAILED'})"
    )

    drift = max(float((p - before[n]).abs().max()) for n, p in named)
    notes.append(
        f"weights restored bit-exactly after both contexts exit: max |dw| {drift:.2e} "
        f"({'OK' if drift == 0.0 else 'FAILED'})"
    )
    return notes


def anomaly_targets(by_month: pd.Series, climatology: pd.Series, target_std: float,
                    block: int = ANOMALY_BLOCK) -> dict[str, float]:
    """The climatological route's target, estimated three ways, in sd of ``target_std``.

    The quantity is the sd of the block-mean observed departure from the 1960-2019 climatology --
    what a long-lead ensemble has to be wide enough to cover, given that the anchor has already
    removed everything else. ``windows`` is the n = 3 estimate from the scored windows themselves,
    ``january`` uses every January-start block in the 65-year record (matching how the windows are
    built, so lead time and season line up), and ``all_starts`` uses every start month. No model
    enters any of them.
    """
    ordered = by_month.sort_index()
    anomaly = np.array([value - climatology[month] for (_year, month), value in ordered.items()])
    starts = np.array([month for _year, month in ordered.index])
    blocks = np.convolve(anomaly, np.ones(block) / block, mode="valid")

    scored = np.array([
        np.mean([by_month[(y, m)] - climatology[m]
                 for y, m in window_months(year, month, steps)[SETTLED_FROM:]])
        for _label, year, month, steps in WINDOWS.values()
    ])
    return {
        "windows": float(scored.std(ddof=1)) / target_std,
        "january": float(blocks[starts[: len(blocks)] == 1].std(ddof=1)) / target_std,
        "all_starts": float(blocks.std(ddof=1)) / target_std,
        "n_january": int((starts[: len(blocks)] == 1).sum()),
    }


def spread_sd(members: np.ndarray, target_std: float) -> float:
    """Across-member sd, pooled over steps and pixels, in standardised units."""
    return float(members.std(axis=0, ddof=1).mean()) / target_std


def score_blocks(members: np.ndarray, truth: np.ndarray, target_std: float) -> dict[str, dict]:
    """Every score on every lead block. ``members`` (M, T, P) already masked; ``truth`` (T, P)."""
    out = {}
    for label, block in ALL_BLOCKS:
        stats = score(members[:, block], truth[block], target_std)
        stats.update(rmse_terms(members[:, block], truth[block]))
        stats["spread_sd"] = spread_sd(members[:, block], target_std)
        out[label] = stats
    return out


def self_anchor(members: np.ndarray, months: list[tuple[int, int]],
                climatology: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Apply the seasonal ramp anchor estimated on THIS ensemble. Returns (shifted, shift).

    This is ``anchor self seas ramp`` from ``utils/climatology_anchor.py``, re-estimated rather than
    reused because the gain moves the rollout the offset is read from. The shift is common to every
    member, so it moves the mean and leaves the across-member spread untouched -- which is precisely
    why the two knobs stay separable.
    """
    pred = members.mean(axis=0).mean(axis=1)  # (T,) masked-mean of the ensemble mean per step
    offset = settled_offset(pred, months, climatology)
    shift = anchor_shift(offset, months, "ramp", seasonal=True)
    return members + shift[:, None], shift


def run_gain(gain: float, seed: int, model: torch.nn.Module, shared: dict, truth: np.ndarray,
             bool_mask: np.ndarray, months: list[tuple[int, int]], climatology: pd.Series,
             target_std: float) -> dict:
    """One rollout at ``gain``, self-anchored, scored on every lead block."""
    with noise_gain(model, gain), dropout_active(model, DROPOUT):
        traj = run_arm(noise_std=1.0, injection="independent", seed=seed, **shared)
    members = traj[:, :, bool_mask]
    shifted, shift = self_anchor(members, months, climatology)
    residual = truth - members.mean(axis=0)
    return {
        "gain": gain,
        "seed": seed,
        "shift": shift,
        "offset_y2_sd": float(shift[SETTLED_FROM:].mean()) / target_std,
        "raw_bias_y2_sd": float(residual[SETTLED_FROM:].mean()) / target_std,
        "spread_by_step": members.std(axis=0, ddof=1).mean(axis=1) / target_std,
        "anchored": score_blocks(shifted, truth, target_std),
        "raw": score_blocks(members, truth, target_std),
    }


def needed_gain(spread_skill: float) -> float:
    """The multiplier a calibrated ensemble would need, 1/ratio. NaN where the ratio is zero.

    The guard is not cosmetic: at g=0 the members are identical, so the ensemble variance is a
    floating-point zero rather than an exact one, and 1/ratio comes back as ~1e14 -- a number that
    would print as a finite answer to a question that has none.
    """
    return 1.0 / spread_skill if spread_skill > 1e-6 else float("nan")


def crossing(gains: np.ndarray, values: np.ndarray, target: float) -> float:
    """Gain at which ``values`` first reaches ``target``, linearly interpolated. NaN if never.

    Used only to read a required gain off the measured curve. It is not an optimisation over the
    test score: the target comes from the observed record (route A) or from the definition of a
    calibrated ratio (route B), and both are fixed before the sweep runs.
    """
    for i in range(1, len(gains)):
        low, high = values[i - 1], values[i]
        if (low - target) * (high - target) <= 0 and high != low:
            return float(gains[i - 1] + (target - low) * (gains[i] - gains[i - 1]) / (high - low))
    return float("nan")


HEAD = ("{:>5s} {:>9s} {:>7s} {:>9s} {:>9s} {:>8s} {:>11s} {:>8s} {:>8s} {:>9s} {:>9s} {:>9s}")
ROW = ("{:>5.2f} {:>9.5f} {:>7.2f} {:>9.5f} {:>9.5f} {:>+8.3f} {:>11.3f} {:>8.3f} {:>7.1f}% "
       "{:>9.4f} {:>9.4f} {:>+9.3f}")


def report_block(rows: list[dict], label: str, note: str, target_spread: float | None) -> None:
    """The sweep on one lead block: width, then what width cost and bought."""
    print(f"\n  {label}   {note}")
    print("  " + HEAD.format("gain", "spread", "x g=1", "CRPS", "CRPS det", "skill", "sprd/skill",
                             "reliab", "outside", "RMSE ens", "RMSE mem", "shift y2"))
    print("  " + "-" * 124)
    base = next(r for r in rows if r["gain"] == REFERENCE_GAIN)["anchored"][label]["spread_sd"]
    best = min(rows, key=lambda r: r["anchored"][label]["crps"])
    for row in rows:
        s = row["anchored"][label]
        marker = " <- CRPS min" if row is best else ""
        print("  " + ROW.format(row["gain"], s["spread_sd"], s["spread_sd"] / base, s["crps"],
                                s["crps_det"], s["crps_skill"], s["spread_skill"], s["reliability"],
                                s["extreme_frac"] * 100, s["rmse_ens"], s["rmse_mem"],
                                row["offset_y2_sd"]) + marker)
    print("  " + "-" * 124)

    gains = np.array([r["gain"] for r in rows])
    ratio = np.array([r["anchored"][label]["spread_skill"] for r in rows])
    spread = np.array([r["anchored"][label]["spread_sd"] for r in rows])
    print(f"  gain reaching spread/skill 1.0        {crossing(gains, ratio, 1.0):>6.2f}")
    if target_spread is not None:
        print(f"  gain reaching spread {target_spread:.3f} sd (route A)  "
              f"{crossing(gains, spread, target_spread):>6.2f}")
    reference = rows[gains.tolist().index(REFERENCE_GAIN)]["anchored"][label]["crps"]
    # An optimum sitting on the last swept gain is not an optimum, it is the edge of the grid. Say
    # so, rather than reporting a bound as if it were a location.
    edge = "  UNBRACKETED: at the edge of the grid, the true minimum is at or beyond it" \
        if best["gain"] == gains[-1] else ""
    print(f"  CRPS minimised at gain                {best['gain']:>6.2f}  "
          f"({(best['anchored'][label]['crps'] / reference - 1) * 100:+.1f}% vs g=1){edge}")


def report_targets(targets: dict[str, float], rows: list[dict], target_std: float) -> float:
    """Print both routes to the required gain, and return route A's target spread in sd."""
    reference = next(r for r in rows if r["gain"] == REFERENCE_GAIN)
    print("\n  THE TARGET, computed before the sweep is read")
    print(f"\n  route A  climatological: sd of the {ANOMALY_BLOCK}-month observed anomaly vs the "
          f"1960-2019 climatology")
    print(f"    {'over the 3 scored windows':38s} {targets['windows']:6.3f} sd   n=3, the figure "
          f"this sweep was commissioned against")
    print(f"    {'over all January-start years':38s} {targets['january']:6.3f} sd   "
          f"n={targets['n_january']}, and the one to aim at")
    print(f"    {'over all start months':38s} {targets['all_starts']:6.3f} sd   overlapping "
          f"blocks, same answer")
    print("    The three scored windows sit at 1.4-1.8 sigma of their own distribution, so the n=3")
    print("    estimate is inflated by a factor of 1.6. Both are carried below; they differ by a")
    print("    full 1.6x in required gain, and no result should be read against the n=3 figure")
    print("    alone.")
    print("\n  route B  spread-skill: 1/ratio, per lead block, at the shipping gain")
    print(f"    {'block':10s} {'spread sd':>11s} {'sprd/skill':>11s} {'needed x':>10s}")
    for label, _block in BLOCKS:
        s = reference["anchored"][label]
        print(f"    {label:10s} {s['spread_sd']:11.5f} {s['spread_skill']:11.3f} "
              f"{needed_gain(s['spread_skill']):10.2f}")
    print("    Not an independent witness: this denominator is the MSE of the anchored ensemble")
    print("    mean, which at long lead is dominated by the same anomaly route A takes the sd of.")
    print("    Nor the same quantity. Route A's target is the sd of the WINDOW-MEAN anomaly, but")
    print("    the spread beside it is the across-member sd at each pixel-month, which must also")
    print("    cover per-pixel and month-to-month departures. Route A is therefore a FLOOR: below")
    print("    it calibration is impossible, above it is not implied. Route B is the criterion.")
    print("    Where the two disagree, that is the structure and not a contradiction.")

    long_lead = reference["anchored"]["m13-24"]
    for name, key in (("65-year", "january"), ("n=3", "windows")):
        needed = targets[key] / long_lead["spread_sd"] if long_lead["spread_sd"] > 0 else np.nan
        verdict = "already clears the floor" if needed <= 1.0 else "below the floor"
        print(f"\n  route A floor at m13-24, {name:8s} target {targets[key]:.3f} sd against "
              f"measured {long_lead['spread_sd']:.3f} sd:  {needed:.2f}x  ({verdict})")
    return targets["january"]


def report_fine(rows: list[dict]) -> None:
    """Whether one global gain can serve both ends of the lead time."""
    print("\n  DOES ONE GAIN SERVE EVERY LEAD?  spread/skill by lead block (1.0 = calibrated)")
    print(f"  {'gain':>5s}" + "".join(f"{label:>10s}" for label, _b in FINE_BLOCKS)
          + f"{'needed x, m1-6':>16s}{'needed x, m19-24':>18s}")
    print("  " + "-" * 79)
    for row in rows:
        ratios = [row["anchored"][label]["spread_skill"] for label, _b in FINE_BLOCKS]
        print(f"  {row['gain']:>5.2f}" + "".join(f"{r:10.3f}" for r in ratios)
              + f"{needed_gain(ratios[0]):16.2f}{needed_gain(ratios[-1]):18.2f}")
    print("  " + "-" * 79)
    print("  If the two 'needed x' columns disagree at g=1, no constant gain can calibrate both")
    print("  ends and the fix is a lead-dependent inflation -- the same shape as the ramp already")
    print("  in the anchor, and buildable from this table without another rollout.")


def report_hist(rows: list[dict], label: str, members: int) -> None:
    """Rank histograms at the shipping gain, the CRPS optimum and the widest gain scored."""
    best = min(rows, key=lambda r: r["anchored"][label]["crps"])
    picked: dict[float, str] = {}
    for gain, name in ((REFERENCE_GAIN, "ships today"), (best["gain"], "CRPS minimum"),
                       (rows[-1]["gain"], "widest scored")):
        # Accumulated rather than assigned: when the CRPS minimum lands on the shipping gain, that
        # coincidence IS the result and the row must not be relabelled into hiding it.
        picked[gain] = f"{picked[gain]} + {name}" if gain in picked else name
    print(f"\n  RANK HISTOGRAMS on {label}  (bin 0 = obs below all members, bin {members} = above)")
    for row in rows:
        if row["gain"] not in picked:
            continue
        h = row["anchored"][label]["rank_hist"]
        bars = "".join("▁▂▃▄▅▆▇█"[min(7, int(v / max(h.max(), 1e-9) * 7.999))] for v in h)
        print(f"    g={row['gain']:<5.2f} {picked[row['gain']]:<28s} {bars}  "
              f"ends {h[0] * 100:5.1f}% /{h[-1] * 100:5.1f}%   "
              f"flat would be {100 / (members + 1):.1f}%   reliab "
              f"{row['anchored'][label]['reliability']:.3f}")
    print("    U-shaped = still too narrow. Dome = overshot. Sloped = the anchor left a location")
    print("    error behind, which would mean the gain is being asked to cover a bias again.")


def report_repeats(repeats: list[dict], rows: list[dict]) -> None:
    """Size the sweep's own noise, so gain-to-gain differences can be read as real or not."""
    print(f"\n  THE SWEEP'S OWN NOISE  ({len(repeats)} seeds at g={REFERENCE_GAIN:.1f}, "
          f"m13-24)")
    for field, fmt in (("spread_sd", "{:.5f}"), ("crps", "{:.5f}"), ("rmse_ens", "{:.4f}")):
        values = np.array([r["anchored"]["m13-24"][field] for r in repeats])
        sd = values.std(ddof=1)
        print(f"    {field:12s} mean {fmt.format(values.mean())}  sd {fmt.format(sd)}"
              f"  ({sd / abs(values.mean()) * 100:.2f}% of the mean)")
    step = np.diff([r["anchored"]["m13-24"]["crps"] for r in rows])
    print(f"    typical gain-to-gain CRPS step on this block: {np.abs(step).mean():.5f}")
    print("    A gain-to-gain difference smaller than the seed sd is not a result.")


def check_reference(rows: list[dict], path: str | None) -> list[str]:
    """Tie g=1 back to the anchor round, so the sweep is known to start where that one ended."""
    if not path or not Path(path).exists():
        return [f"g=1 vs the anchor round: SKIPPED (no reference npz at {path!r})"]
    d = np.load(path, allow_pickle=True)
    key = "on" if DROPOUT else "off"
    arms = [str(a) for a in d[f"{key}_arms"]]
    idx = arms.index("anchor self  seas  ramp")
    reference = next(r for r in rows if r["gain"] == REFERENCE_GAIN)["anchored"]["m1-24"]
    notes = []
    for field in ("crps", "spread_skill", "rmse_ens"):
        want, got = float(d[f"{key}_{field}"][idx]), reference[field]
        rel = abs(got - want) / abs(want)
        notes.append(
            f"g=1 reproduces {Path(path).name} {field}: {want:.5f} vs {got:.5f}, "
            f"{rel * 100:.2f}% ({'OK' if rel < 0.02 else 'FAILED'})"
        )
    return notes


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Sweep the injection gain over the anchored test ensemble and score it lead-resolved."""
    label, year, month, steps = WINDOWS[SCORED_WINDOW]
    cfg = copy.deepcopy(cfg)
    cfg["dataset"]["starting_year"] = year
    cfg["dataset"]["starting_month"] = month

    climatology, by_month = observed_climatology(cfg)
    months = window_months(year, month, steps)

    # One ``prepare`` for the whole sweep: loading and rasterising the parquet dominates the cost of
    # a 24-step rollout, and every gain runs on identical inputs by construction this way.
    prepared = prepare(cfg)
    model = prepared["model"]
    shared = dict(prepared["rollout"])
    if prepared["num_steps"] < steps:
        raise ValueError(f"window {year}-{month:02d} has {prepared['num_steps']} steps, "
                         f"{steps} requested")
    shared["feature_maps"] = shared["feature_maps"][:steps]
    shared["timestamps"] = shared["timestamps"][:steps]

    bool_mask, target_std = prepared["bool_mask"], prepared["target_std"]
    truth = prepared["y_true"][:steps][:, bool_mask]
    num_members = prepared["num_members"]

    notes = check_gain(model)

    rows = []
    for gain in GAINS:
        rows.append(run_gain(gain, 0, model, shared, truth, bool_mask, months, climatology,
                             target_std))
        s = rows[-1]["anchored"]["m13-24"]
        logger.info(f"g={gain:<5.2f} m13-24 spread={s['spread_sd']:.5f} crps={s['crps']:.5f} "
                    f"rmse_ens={s['rmse_ens']:.4f} sprd/skill={s['spread_skill']:.3f}")
    repeats = [rows[GAINS.index(REFERENCE_GAIN)]] + [
        run_gain(REFERENCE_GAIN, seed, model, shared, truth, bool_mask, months, climatology,
                 target_std)
        for seed in REPEAT_SEEDS[1:]
    ]

    reference_path = (cfg["model"]["reference_anchor_path"]
                      if "reference_anchor_path" in cfg["model"] else None)
    notes += check_reference(rows, reference_path)

    run = Path(cfg["model"]["checkpoint_path"]).parts[0]
    print("\n" + "=" * 128)
    print("NOISE-GAIN SWEEP OVER THE ANCHORED ENSEMBLE".center(128))
    print(f"{run} · {num_members} members · {steps} steps · dropout "
          f"{'ON' if DROPOUT else 'off'} · scored on {label}".center(128))
    print("anchor self seas ramp, RE-ESTIMATED at every gain · CRPS and spreads in "
          "standardised units".center(128))
    print("=" * 128)
    for note in notes:
        print(f"  check  {note}")

    targets = anomaly_targets(by_month, climatology, target_std)
    target_spread = report_targets(targets, rows, target_std)

    report_block(rows, "m1-24", "the whole window, for continuity with the anchor round", None)
    report_block(rows, "m1-12", "IC still live: the ensemble is SUPPOSED to be narrow here", None)
    report_block(rows, "m13-24", "settled: what is left is the climate anomaly", target_spread)

    report_fine(rows)
    report_hist(rows, "m13-24", num_members)
    report_repeats(repeats, rows)

    print("\n  WHAT THE WIDTH COST THE MEAN  (m13-24, against the g=1 row)")
    ref = rows[GAINS.index(REFERENCE_GAIN)]["anchored"]["m13-24"]
    print(f"  {'gain':>5s} {'spread x':>10s} {'RMSE ens':>10s} {'RMSE %':>9s} {'CRPS %':>9s} "
          f"{'raw bias y2':>12s} {'shift y2':>10s}")
    print("  " + "-" * 72)
    for row in rows:
        s = row["anchored"]["m13-24"]
        print(f"  {row['gain']:>5.2f} {s['spread_sd'] / ref['spread_sd']:10.2f} "
              f"{s['rmse_ens']:10.4f} {(s['rmse_ens'] / ref['rmse_ens'] - 1) * 100:+8.1f}% "
              f"{(s['crps'] / ref['crps'] - 1) * 100:+8.1f}% {row['raw_bias_y2_sd']:+12.3f} "
              f"{row['offset_y2_sd']:+10.3f}")
    print("  " + "-" * 72)
    print("  g=0 is injection_off: the width it loses is the whole ensemble, and the RMSE column")
    print("  there is the standing measure of how load-bearing the injection is. If RMSE rises")
    print("  faster than spread on the way up, the gain is buying width by breaking the forecast,")
    print("  and CRPS -- being proper -- will say so before the spread column does.")
    print("  raw bias y2 moving with the gain is the mean drifting under it; shift y2 is the")
    print("  anchor absorbing that drift, which is why it is re-estimated at every point.")
    print("=" * 128 + "\n")

    out = cfg["model"].get("saving_path")
    if not out:
        return
    saved: dict[str, np.ndarray] = {
        "gains": np.array([r["gain"] for r in rows]),
        "blocks": np.array([label for label, _b in ALL_BLOCKS]),
        "target_std": np.array(target_std),
        "shift": np.stack([r["shift"] for r in rows]),
        "spread_by_step": np.stack([r["spread_by_step"] for r in rows]),
        "offset_y2_sd": np.array([r["offset_y2_sd"] for r in rows]),
        "raw_bias_y2_sd": np.array([r["raw_bias_y2_sd"] for r in rows]),
        "repeat_seeds": np.array(REPEAT_SEEDS),
        "noise_weights_rms": np.array(weights_rms(model)),
        **{f"target_{k}": np.array(v) for k, v in targets.items()},
    }
    for state in ("anchored", "raw"):
        for block, _slice in ALL_BLOCKS:
            for field in ("spread_sd", "crps", "crps_det", "crps_skill", "spread_skill",
                          "reliability", "extreme_frac", "coverage", "rmse_ens", "rmse_mem",
                          "rmse_max"):
                saved[f"{state}_{block}_{field}"] = np.array(
                    [r[state][block][field] for r in rows])
            saved[f"{state}_{block}_rank_hist"] = np.stack(
                [r[state][block]["rank_hist"] for r in rows])
    for field in ("spread_sd", "crps", "rmse_ens"):
        saved[f"repeat_m13-24_{field}"] = np.array(
            [r["anchored"]["m13-24"][field] for r in repeats])
    np.savez(f"{out}_gain.npz", **saved)
    logger.info(f"Saved to '{out}_gain.npz'.")


if __name__ == "__main__":
    main()
