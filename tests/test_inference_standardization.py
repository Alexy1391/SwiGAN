"""Tests that the inference path feeds the SWI history in the unit system training used.

The generator's last ``num_input_steps`` input channels are SWI frames. The trainer hands them
over STANDARDIZED -- ``utils.swi_dataset`` applies ``np.where(mask, (x - mean) / std, 0.0)`` to
every split -- while ``utils.preprocessing.dataframe_to_rasters`` returns RAW SWI. Every
free-running inference path seeds its rollout from a dataframe, so it has to bridge that gap
itself; for a long time it did not, and the first ``num_input_steps`` months of every rollout
went in raw. On Corse that was a +0.90 sigma offset at 0.385x contrast.

These tests pin the bridge: the seed the inference path builds must be bit-identical to the
history ``SWIDataset`` hands the trainer for the same months.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.preprocessing import single_channel_statistic, standardize_swi_history
from utils.swi_dataset import build_datasets_from_split_rasters

NUM_INPUT_STEPS = 8
NUM_FEATURES = 11
HEIGHT, WIDTH = 23, 11  # the Corse grid


def _corse_like_rasters(
    num_months: int = 40, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic rasters on a Corse-shaped grid with a Corse-like SWI annual cycle."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    mask[:, 2:21, 3:9] = 1.0  # an island, so most of the grid is padding

    months = np.arange(num_months) % 12
    # Corse dries to the wilting point every August, which is what makes the offset bite.
    cycle = 0.48 + 0.45 * np.cos(2 * np.pi * (months - 1) / 12)
    targets = (
        cycle[:, None, None, None] + 0.05 * rng.standard_normal((num_months, 1, HEIGHT, WIDTH))
    ).astype(np.float32)
    feats = rng.standard_normal((num_months, NUM_FEATURES, HEIGHT, WIDTH)).astype(np.float32)
    timesteps = months.astype(np.float32).reshape(-1, 1)
    return feats, targets, timesteps, mask


def _trainer_history(index: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build the SWI history the trainer feeds for sample ``index``, with the raw maps."""
    feats, targets, timesteps, mask = _corse_like_rasters()
    datasets, statistics = build_datasets_from_split_rasters(
        split_maps={"train": feats},
        split_targets={"train": targets},
        split_timesteps={"train": timesteps},
        mask=mask,
        num_input_steps=NUM_INPUT_STEPS,
    )
    sample = datasets["train"][index]
    history = sample["input_maps"][NUM_FEATURES:].numpy()
    raw_seed = targets[index : index + NUM_INPUT_STEPS].squeeze(1)
    return history, raw_seed, {**statistics, "mask": mask}


def test_inference_seed_matches_the_trainers_history() -> None:
    """The bridge is exact: same months in, same numbers out, to float32 precision."""
    history, raw_seed, stats = _trainer_history(index=0)
    seed = standardize_swi_history(raw_seed, stats, stats["mask"])
    np.testing.assert_allclose(seed, history, rtol=0, atol=1e-6)


def test_raw_history_does_not_match_and_is_biased_wet() -> None:
    """Guard the regression itself: feeding raw SWI is materially wrong, not a rounding detail.

    If someone drops the ``standardize_swi_history`` call, the first test fails and this one
    records why -- the seed lands almost a standard deviation wet, at a third of its contrast.
    """
    history, raw_seed, stats = _trainer_history(index=0)
    region = stats["mask"].squeeze().astype(bool)
    std = single_channel_statistic(stats, "targets_std")

    mean = single_channel_statistic(stats, "targets_mean")
    raw, correct = raw_seed[:, region], history[:, region]

    assert not np.allclose(raw, correct, atol=1e-3)

    # The damage is affine and exact: a cell the model should read as ``z`` it reads as
    # ``z * std + mean`` instead. Pin the identity rather than a magic threshold.
    np.testing.assert_allclose(raw, correct * std + mean, rtol=0, atol=1e-6)

    # Which cashes out as a seed biased wet at a fraction of its true contrast.
    assert raw.mean() - correct.mean() > 0.25, (
        "raw SWI should read wet against a standardized history"
    )
    assert raw.std() / correct.std() == pytest.approx(std, rel=0.05)


def test_padding_reads_as_the_training_mean_not_a_shifted_zero() -> None:
    """Standardize first, THEN zero the padding -- the order the trainer uses.

    Masking before standardizing would leave every padded cell at ``-mean / std``, an
    out-of-range constant that the next 3x3 convolution bleeds back across the coast.
    """
    history, raw_seed, stats = _trainer_history(index=0)
    seed = standardize_swi_history(raw_seed, stats, stats["mask"])
    padding = ~stats["mask"].squeeze().astype(bool)
    assert np.all(seed[:, padding] == 0.0)


def test_seed_keeps_its_shape() -> None:
    """The statistics are stored with ``keepdims=True``; used raw they would add an axis."""
    _, raw_seed, stats = _trainer_history(index=0)
    assert stats["targets_mean"].shape == (1, 1, 1, 1)
    seed = standardize_swi_history(raw_seed, stats, stats["mask"])
    assert seed.shape == (NUM_INPUT_STEPS, HEIGHT, WIDTH)


def test_single_channel_statistic_rejects_a_multi_channel_target() -> None:
    """It is only a scalar because the generator has one output channel. Say so if that changes."""
    with pytest.raises(ValueError, match="not single-channel"):
        single_channel_statistic({"targets_mean": np.zeros((1, 3, 1, 1))}, "targets_mean")
