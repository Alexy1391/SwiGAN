"""The state-dispersion instrument: what an ensemble member sees, and the seasonal offset.

WHY THIS EXISTS. The shipped Corse ensemble is 20 pixel-scale variations around one regional
trajectory: region-mean spread 0.011 SWI in every month against a required 0.04-0.18. The
injection gain (``utils/noise_gain_sweep.py``) can widen it only by pushing every member wet at
low SWI, which is how it reaches calibration at +22% CRPS and +22% on the ensemble mean's RMSE.

The two knobs here perturb the STATE the generator conditions on instead of the noise inside
it, and leave its own feedback untouched:

  neutral members    k of the M members see a climatologically neutral SWI history
                     (``history * 0``, i.e. the training mean in standardised units). This is
                     the neutral-history control of the 250-epoch report promoted to an ensemble
                     component. It corrects the model's over-persistence -- the ensemble mean's
                     RMSE falls about 5% on every window tried -- and is what puts members on a
                     regime break such as October 2024.
  state perturbation the M members see the history shifted by a fixed b_m ~ N(0, sd) in
                     standardised units: an initial-condition perturbation, coherent over the
                     region and persistent down the rollout. This is the only single knob that
                     reaches calibration (spread-skill 0.94 at sd 1.0), at +3% on the mean.

Together (four neutral members, sd 0.6-1.0) they beat the shipped ensemble on CRPS and on the
mean's RMSE on 7 of 7 windows at spread-skill 0.88-1.12, where the gain paid 22% for the same
width. See ``outputs/2026-09-16/arm_comparison/`` and the report section "Widen the state,
not the noise".

Both transforms are applied to the INPUT COPY of the history buffer at every step -- see
``swigan.inference.compute_trajectory_iterative``'s ``history_scale`` / ``history_shift`` --
so a member's own predictions are never rescaled twice.

THE SEASONAL OFFSET. Neither knob moves the bias (mean |bias| 0.049 -> 0.047-0.054 over seven
windows). A per-calendar-month offset fitted on past OBSERVED windows takes about 3% CRPS on the
shipped ensemble and about 5% on top of the state-dispersed one; a single global scalar takes
nothing, which is the climatology anchor's Corse result restated with an estimator that is
allowed to see observations. ``fit_seasonal_offsets`` / ``apply_seasonal_offsets`` are that
correction. It is fitted on windows that end before the forecast starts, never on the scored
window: ``utils/dispersion_select.py`` does the selection out of sample.

What none of this does is make the width CONDITIONAL on the state -- a fixed perturbation is a
fixed width, twice too wide in August and too narrow in November. That is the training-side
objective's job (``loss_fn: energy`` in the trainer), and the per-month test of
:mod:`utils.spread_shape` is what tells the two apart.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_MEMBER_SEED = 0


def member_perturbations(
    num_members: int, neutral_members: int, state_sd: float, seed: int = DEFAULT_MEMBER_SEED
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Per-member ``(scale, shift)`` of the history each member sees, on the CPU.

    Members ``0 .. neutral_members-1`` get scale 0 (neutral history), the rest scale 1; every
    member gets a shift drawn from ``N(0, state_sd)`` on a generator seeded with ``seed``, so
    the same ``(num_members, state_sd, seed)`` always builds the same ensemble. Which members
    are neutral is immaterial -- members differ only by their noise draws.

    Returns ``(None, None)`` when both knobs are off, which is the signal to
    ``compute_trajectory_iterative`` to take its unperturbed, bit-exact path.
    """
    if not 0 <= neutral_members <= num_members:
        raise ValueError(f"neutral_members must be in 0..{num_members}, got {neutral_members}")
    if state_sd < 0:
        raise ValueError(f"state_sd must be non-negative, got {state_sd}")
    if neutral_members == 0 and state_sd == 0:
        return None, None
    scale = torch.ones(num_members)
    scale[:neutral_members] = 0.0
    generator = torch.Generator().manual_seed(int(seed))
    shift = torch.randn(num_members, generator=generator) * float(state_sd)
    return scale, shift


def month_index(timestamps: np.ndarray) -> np.ndarray:
    """Read the 0-based calendar month of each step, as ``dataframe_to_rasters`` encodes it."""
    months = np.asarray(timestamps).reshape(len(np.asarray(timestamps)), -1)[:, 0]
    months = np.rint(months).astype(int)
    if months.min() < 0 or months.max() > 11:
        raise ValueError("timestamps must encode the month as 0..11")
    return months


def fit_seasonal_offsets(
    ensemble_means: list[np.ndarray], truths: list[np.ndarray], months: list[np.ndarray]
) -> np.ndarray:
    """Twelve offsets, ``mean(ensemble mean - observation)`` per calendar month, pooled.

    Each entry of the three lists is one window: ``(T, ...)`` ensemble mean and observation
    over the region's cells only, and the ``(T,)`` month index of its steps. Pooled over every
    cell-month of every window, so a window contributes in proportion to what it holds. A
    month no window reaches gets an offset of 0 -- the correction leaves it alone.
    """
    if not len(ensemble_means) == len(truths) == len(months) > 0:
        raise ValueError("ensemble_means, truths and months must be non-empty and aligned")
    total = np.zeros(12)
    count = np.zeros(12)
    for ens, obs, mon in zip(ensemble_means, truths, months, strict=True):
        ens = np.asarray(ens, dtype=np.float64)
        obs = np.asarray(obs, dtype=np.float64)
        mon = np.asarray(mon).reshape(-1).astype(int)
        if ens.shape != obs.shape or ens.shape[0] != len(mon):
            raise ValueError("each window's ensemble mean, observation and months must align")
        err = (ens - obs).reshape(len(mon), -1)
        for m in range(12):
            rows = err[mon == m]
            total[m] += rows.sum()
            count[m] += rows.size
    with np.errstate(invalid="ignore", divide="ignore"):
        offsets = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    return offsets


def apply_seasonal_offsets(
    members: np.ndarray,
    months: np.ndarray,
    offsets: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Subtract ``offsets[month]`` from every member at every step, on the region only.

    ``members`` is ``(M, T, ...)`` with the time axis second; ``mask`` broadcasts against the
    trailing dimensions (``(1, H, W)`` for maps, ``(P,)`` for flattened cells), so the padding
    stays at exactly zero. Works in whatever units ``offsets`` were fitted in.
    """
    members = np.asarray(members)
    months = np.asarray(months).reshape(-1).astype(int)
    offsets = np.asarray(offsets, dtype=members.dtype).reshape(-1)
    if offsets.shape != (12,):
        raise ValueError(f"offsets must hold 12 monthly values, got {offsets.shape}")
    if members.ndim < 2 or members.shape[1] != len(months):
        raise ValueError("members must be (M, T, ...) with one month per step")
    shift = offsets[months].reshape((1, len(months)) + (1,) * (members.ndim - 2))
    if mask is not None:
        mask = np.asarray(mask, dtype=members.dtype)
        mask = mask.reshape((1,) * (members.ndim - mask.ndim) + mask.shape)
        shift = shift * mask
    return members - shift


def dispersion_spec(
    neutral_members: int,
    state_sd: float,
    seasonal_offsets: np.ndarray | None,
    member_seed: int = DEFAULT_MEMBER_SEED,
    **provenance: Any,
) -> dict[str, Any]:
    """Build the dictionary ``swigan/inference.py`` consumes, JSON-serialisable."""
    return {
        "neutral_members": int(neutral_members),
        "state_sd": float(state_sd),
        "member_seed": int(member_seed),
        "seasonal_offsets": None
        if seasonal_offsets is None
        else [float(v) for v in np.asarray(seasonal_offsets).reshape(-1)],
        **provenance,
    }


def load_dispersion_spec(source: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Read a spec from a JSON path or pass a dict through, validating the fields used."""
    spec = dict(source) if isinstance(source, dict) else json.loads(Path(source).read_text())
    for key in ("neutral_members", "state_sd"):
        if key not in spec:
            raise ValueError(f"dispersion spec is missing '{key}'")
    spec.setdefault("member_seed", DEFAULT_MEMBER_SEED)
    spec.setdefault("seasonal_offsets", None)
    if spec["seasonal_offsets"] is not None and len(spec["seasonal_offsets"]) != 12:
        raise ValueError("seasonal_offsets must hold 12 values or be null")
    return spec


def save_dispersion_spec(spec: dict[str, Any], path: str | Path) -> None:
    """Write the spec as indented JSON."""
    Path(path).write_text(json.dumps(spec, indent=1))
