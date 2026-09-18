"""Tests of the per-month spread-shape test and of the ensemble scores on the rollout path.

The annual spread-skill ratio cannot see an ensemble that is too wide in one season and too
narrow in another; the monthly ratio of :mod:`utils.spread_shape` exists for that. These tests
pin that a calibrated ensemble passes it month by month, that a flat-spread ensemble against a
seasonal error fails it in the predicted months, that the region level tells coherent spread
from pixel speckle, and that pooling windows through the accumulator is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.ensemble_scores import score
from utils.rollout import summarise_rollout
from utils.spread_shape import (
    BAND,
    MonthlySpreadAccumulator,
    format_monthly_table,
    monthly_spread_ratio,
    spread_shape_verdict,
)

M, YEARS, P = 20, 30, 40
# The Corse shape: uncertainty four times larger in November than in August.
SEASONAL_SD = np.array([0.11, 0.15, 0.15, 0.15, 0.13, 0.15, 0.07, 0.04, 0.05, 0.14, 0.18, 0.14])


def _months(years: int = YEARS) -> np.ndarray:
    return np.arange(12 * years) % 12


def _calibrated(
    seed: int = 0, level_sd: np.ndarray = SEASONAL_SD, years: int = YEARS
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the truth from the same distribution the members are drawn from, month by month."""
    rng = np.random.default_rng(seed)
    months = _months(years)
    sd = level_sd[months][:, None]  # (T, 1)
    mu = rng.normal(0.5, 0.2, size=(len(months), P))
    truth = mu + sd * rng.standard_normal((len(months), P))
    members = mu[None] + sd[None] * rng.standard_normal((M, len(months), P))
    return members, truth, months


def test_calibrated_ensemble_passes_every_month() -> None:
    """An ensemble drawn from the truth's distribution reads 1.0 in every month."""
    members, truth, months = _calibrated()
    ratios = monthly_spread_ratio(members, truth, months, level="pixel")
    assert ratios.shape == (12,)
    assert np.all(np.abs(ratios - 1.0) < 0.06), ratios
    assert spread_shape_verdict(ratios)["pass"]


def test_flat_spread_fails_in_the_predicted_months() -> None:
    """A fixed spread against a seasonal error is too wide in summer and too narrow in autumn."""
    rng = np.random.default_rng(1)
    months = _months()
    mu = rng.normal(0.5, 0.2, size=(len(months), P))
    truth = mu + SEASONAL_SD[months][:, None] * rng.standard_normal((len(months), P))
    flat = 0.10
    members = mu[None] + flat * rng.standard_normal((M, len(months), P))
    ratios = monthly_spread_ratio(members, truth, months, level="pixel")
    # produced carries the (M+1)/M correction; required is the truth's sd plus the ensemble
    # mean's own sampling noise, flat / sqrt(M). For a calibrated ensemble the two cancel
    # exactly, which is what the correction is for.
    expected = np.sqrt((M + 1) / M) * flat / np.sqrt(SEASONAL_SD**2 + flat**2 / M)
    assert np.allclose(ratios, expected, rtol=0.06), (ratios, expected)
    verdict = spread_shape_verdict(ratios)
    assert not verdict["pass"]
    assert ratios[7] > BAND[1]  # August: twice too wide
    assert ratios[10] < BAND[0]  # November: too narrow
    assert verdict["worst_month"] == "Aug"


def test_region_level_sees_through_pixel_speckle() -> None:
    """Independent per-pixel noise widens the pixel ratio but averages out of the region mean."""
    rng = np.random.default_rng(2)
    months = _months(years=300)  # region level has one sample per window-month
    steps = len(months)
    mu = rng.normal(0.5, 0.2, size=(steps, 1)) * np.ones((1, P))
    # The truth moves coherently over the region.
    truth = mu + 0.1 * rng.standard_normal((steps, 1))
    speckle = mu[None] + 0.1 * rng.standard_normal((M, steps, P))
    coherent = mu[None] + 0.1 * rng.standard_normal((M, steps, 1)) * np.ones((1, 1, P))

    pix_speckle = monthly_spread_ratio(speckle, truth, months, "pixel")
    reg_speckle = monthly_spread_ratio(speckle, truth, months, "region")
    reg_coherent = monthly_spread_ratio(coherent, truth, months, "region")

    assert np.all(np.abs(pix_speckle - 1.0) < 0.1)  # calibrated cell by cell ...
    assert np.all(reg_speckle < 0.3)  # ... and nearly no spread on the region mean
    assert np.all(np.abs(reg_coherent - 1.0) < 0.12)  # coherent spread is what the region needs


def test_accumulator_pools_windows_exactly() -> None:
    """Two windows through the accumulator equal one window of their concatenation."""
    members, truth, months = _calibrated(seed=3, years=6)
    cut = 12 * 2 + 5  # an uneven split, so the two windows do not hold the same months
    pooled = MonthlySpreadAccumulator("region")
    pooled.add(members[:, :cut], truth[:cut], months[:cut])
    pooled.add(members[:, cut:], truth[cut:], months[cut:])
    single = monthly_spread_ratio(members, truth, months, "region")
    assert np.allclose(pooled.ratios(), single)
    assert pooled.count.sum() == len(months)


def test_accumulator_rejects_inconsistent_windows() -> None:
    """Member counts, step counts and the level are all checked."""
    members, truth, months = _calibrated(seed=4, years=2)
    acc = MonthlySpreadAccumulator("pixel")
    acc.add(members, truth, months)
    with pytest.raises(ValueError, match="same number of members"):
        acc.add(members[:5], truth, months)
    with pytest.raises(ValueError, match="one entry per step"):
        acc.add(members, truth, months[:-1])
    with pytest.raises(ValueError, match="at least 2 members"):
        MonthlySpreadAccumulator("pixel").add(members[:1], truth, months)
    with pytest.raises(ValueError, match="level must be"):
        MonthlySpreadAccumulator("field")


def test_verdict_ignores_missing_months_and_validates_band() -> None:
    """Months without samples are neither passes nor failures; the band must be ordered."""
    ratios = np.full(12, np.nan)
    ratios[[0, 5, 9]] = [1.0, 0.9, 1.2]
    verdict = spread_shape_verdict(ratios)
    assert verdict["n_months"] == 3
    assert verdict["n_in_band"] == 3
    assert verdict["pass"]
    ratios[9] = 1.5
    verdict = spread_shape_verdict(ratios)
    assert verdict["n_in_band"] == 2
    assert not verdict["pass"]
    assert verdict["worst_month"] == "Oct"
    assert not spread_shape_verdict(np.full(12, np.nan))["pass"]
    with pytest.raises(ValueError, match="band"):
        spread_shape_verdict(np.ones(12), band=(1.2, 0.8))
    with pytest.raises(ValueError, match="12 monthly"):
        spread_shape_verdict(np.ones(11))
    table = format_monthly_table({"ratio": ratios}, header="arm")
    assert "Oct" in table
    assert " -- " in table


def test_rollout_summary_adds_ensemble_scores_consistent_with_score() -> None:
    """The training-time scores are the offline ``score`` on the same masked members."""
    rng = np.random.default_rng(5)
    members_n, steps, height, width = 8, 24, 6, 5
    mask = np.zeros((height, width), dtype=bool)
    mask[1:5, 1:4] = True
    months = np.arange(steps) % 12
    y_true = rng.normal(0.5, 0.2, size=(steps, height, width)) * mask
    noise = 0.1 * rng.standard_normal((members_n, steps, height, width))
    trajectories = (y_true[None] + noise) * mask
    target_std = 0.3846

    summary = summarise_rollout(trajectories, y_true, mask, months=months, target_std=target_std)
    reference = score(trajectories[:, :, mask], y_true[:, mask], target_std)
    assert summary["rollout_crps"] == pytest.approx(reference["crps"])
    assert summary["rollout_spread_skill"] == pytest.approx(reference["spread_skill"])
    assert summary["spread_ratio_by_month"].shape == (12,)
    assert 0.0 <= summary["rollout_spread_shape_in_band"] <= 1.0
    assert summary["rollout_spread_shape_min"] <= summary["rollout_spread_shape_max"]
    # The pre-existing keys are untouched.
    for key in ("rollout_rmse", "rollout_rmse_max", "rollout_drift", "rmse_by_step"):
        assert key in summary

    without_months = summarise_rollout(trajectories, y_true, mask)
    assert "rollout_crps" in without_months
    assert "rollout_spread_shape_min" not in without_months
    single = summarise_rollout(trajectories[:1], y_true, mask, months=months)
    assert "rollout_crps" not in single
    assert "rollout_rmse" in single
