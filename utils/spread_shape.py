"""The acceptance test for dispersion: whether the spread has the SHAPE the errors have.

WHY THIS EXISTS. The annual spread-skill ratio -- sqrt(mean ensemble variance / MSE of the
ensemble mean), 1.0 when calibrated -- is what every round so far was judged on. On Corse the
mixture-plus-state-perturbation ensemble passes it at 1.118 while being TWICE too wide in
July-August and too narrow in November-December: the real uncertainty of the region mean
swings fourfold over the year (0.042 SWI in August, 0.179 in November) and a fixed inference
perturbation produces a nearly flat 0.09-0.18. One number over the year cannot see that. The
same ratio taken per calendar month can, and it is what a training objective that credits
conditional spread has to be held to.

WHAT IS COMPUTED. For each calendar month, over every (window-year, cell) that falls in it:

    produced   sqrt( (M+1)/M * mean over samples of the across-member variance )
    required   sqrt( mean over samples of (ensemble mean - observation)^2 )
    ratio      produced / required

so the ratio is the spread-skill ratio restricted to one month, with the finite-ensemble
correction ``score`` in :mod:`utils.ensemble_scores` uses. Two levels:

    region   samples are window-months; the variance and the error are those of the REGION
             MEAN. This is the quantity the insurance loss distribution of the paper's
             Section 6 consumes, and the one the fixed knobs get wrong.
    pixel    samples are pixel-months. Many more of them, so it is readable on a single
             24-month window and is what the training callback logs every epoch.

THE TEST. Every month's ratio inside ``BAND`` = (0.8, 1.25) -- a factor of 1.25 either way,
symmetric on a log scale. An ensemble at the right annual width that fails this is putting
its spread in the wrong months, which for a tail quantity is the same as having none.

POOLING. Ratios over several windows are pooled from the sums, not averaged from per-window
ratios: :class:`MonthlySpreadAccumulator` keeps the variance and squared-error totals per
month and forms the ratio once at the end, so a window with two Octobers and a window with
none weigh what they hold.
"""

from __future__ import annotations

import numpy as np

BAND: tuple[float, float] = (0.8, 1.25)
LEVELS: tuple[str, ...] = ("region", "pixel")
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _as_month_index(months: np.ndarray) -> np.ndarray:
    months = np.asarray(months).reshape(-1).astype(int)
    if months.min() < 0 or months.max() > 11:
        raise ValueError("months must be 0-based calendar month indices in 0..11")
    return months


def _flatten_cells(members: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(M, T, ...)`` and ``(T, ...)`` to ``(M, T, P)`` and ``(T, P)``."""
    members = np.asarray(members, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if members.ndim < 3 or truth.ndim < 2 or members.shape[1:] != truth.shape:
        raise ValueError(
            f"members must be (M, T, ...) and truth (T, ...) with matching trailing shape, "
            f"got {members.shape} and {truth.shape}"
        )
    flat_members = members.reshape(members.shape[0], members.shape[1], -1)
    return flat_members, truth.reshape(truth.shape[0], -1)


class MonthlySpreadAccumulator:
    """Per-calendar-month sums of ensemble variance and squared error, poolable over windows.

    ``add`` takes one window's members ``(M, T, ...)`` in the same units as ``truth`` ``(T, ...)``
    and the ``(T,)`` 0-based month index of each step; cells outside the region should already
    be dropped (pass ``members[:, :, mask]``), because a zeroed padding cell would count as a
    perfectly predicted, zero-spread sample. ``ratios`` returns the twelve monthly ratios and
    NaN for a month no window reached.
    """

    def __init__(self, level: str = "region") -> None:
        """Start empty at one ``level``; ``add`` windows, then read ``ratios``."""
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.level = level
        self.num_members: int | None = None
        self.var_sum = np.zeros(12)
        self.err_sum = np.zeros(12)
        self.count = np.zeros(12)

    def add(self, members: np.ndarray, truth: np.ndarray, months: np.ndarray) -> None:
        """Accumulate one window."""
        members, truth = _flatten_cells(members, truth)
        months = _as_month_index(months)
        if len(months) != members.shape[1]:
            raise ValueError("months must have one entry per step")
        num_members = members.shape[0]
        if num_members < 2:
            raise ValueError("at least 2 members are needed for a spread")
        if self.num_members is None:
            self.num_members = num_members
        elif self.num_members != num_members:
            raise ValueError("every window must carry the same number of members")

        if self.level == "region":
            reg = members.mean(axis=-1)  # (M, T)
            var = reg.var(axis=0, ddof=1)  # (T,)
            err = (reg.mean(axis=0) - truth.mean(axis=-1)) ** 2  # (T,)
        else:
            var = members.var(axis=0, ddof=1)  # (T, P)
            err = (members.mean(axis=0) - truth) ** 2  # (T, P)
        for step, month in enumerate(months):
            self.var_sum[month] += float(np.sum(var[step]))
            self.err_sum[month] += float(np.sum(err[step]))
            self.count[month] += float(np.size(var[step]))

    def produced(self) -> np.ndarray:
        """Spread per month, with the finite-ensemble correction."""
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = (self.num_members + 1) / self.num_members if self.num_members else np.nan
            return np.sqrt(factor * self.var_sum / self.count)

    def required(self) -> np.ndarray:
        """RMS error of the ensemble mean per month: the spread a calibrated ensemble needs."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(self.err_sum / self.count)

    def ratios(self) -> np.ndarray:
        """Twelve monthly spread / required ratios, NaN where a month has no samples."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.produced() / self.required()


def monthly_spread_ratio(
    members: np.ndarray, truth: np.ndarray, months: np.ndarray, level: str = "region"
) -> np.ndarray:
    """Per-month ratios of one window. See :class:`MonthlySpreadAccumulator`."""
    acc = MonthlySpreadAccumulator(level)
    acc.add(members, truth, months)
    return acc.ratios()


def spread_shape_verdict(ratios: np.ndarray, band: tuple[float, float] = BAND) -> dict:
    """Pass/fail of the twelve ratios against ``band``, ignoring months without samples."""
    ratios = np.asarray(ratios, dtype=np.float64).reshape(-1)
    if ratios.shape != (12,):
        raise ValueError(f"expected 12 monthly ratios, got shape {ratios.shape}")
    lo, hi = band
    if not 0 < lo < hi:
        raise ValueError(f"band must satisfy 0 < lo < hi, got {band}")
    present = ~np.isnan(ratios)
    inside = present & (ratios >= lo) & (ratios <= hi)
    n_present = int(present.sum())
    return {
        "band": (float(lo), float(hi)),
        "n_months": n_present,
        "n_in_band": int(inside.sum()),
        "in_band_fraction": float(inside.sum() / n_present) if n_present else float("nan"),
        "min": float(np.nanmin(ratios)) if n_present else float("nan"),
        "max": float(np.nanmax(ratios)) if n_present else float("nan"),
        "worst_month": MONTH_LABELS[int(np.nanargmax(np.abs(np.log(ratios))))]
        if n_present
        else None,
        "pass": bool(n_present > 0 and inside.sum() == n_present),
    }


def format_monthly_table(rows: dict[str, np.ndarray], header: str = "") -> str:
    """One line per named series of twelve monthly values, for the logs."""
    lines = [f"{header:<28s}" + " ".join(f"{m:>6s}" for m in MONTH_LABELS)]
    for name, values in rows.items():
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        cells = " ".join("   -- " if np.isnan(v) else f"{v:6.3f}" for v in values)
        lines.append(f"{name:<28s}{cells}")
    return "\n".join(lines)
