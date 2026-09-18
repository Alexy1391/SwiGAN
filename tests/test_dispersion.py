"""Tests of the state-dispersion instrument and of the perturbed rollout path.

The rollout's ``history_scale`` / ``history_shift`` transform what the generator SEES and must
leave what it feeds back untouched; with both off the path has to be the original one bit for
bit, because it is what ships. The seasonal offset has to recover a known monthly bias and stay
off the padding. These tests pin those properties on a stand-in generator whose output is a
known function of its history channels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from swigan.inference import compute_trajectory_iterative
from utils.dispersion import (
    apply_seasonal_offsets,
    dispersion_spec,
    fit_seasonal_offsets,
    load_dispersion_spec,
    member_perturbations,
    month_index,
    save_dispersion_spec,
)

HISTORY, HEIGHT, WIDTH, FEATURES, Z_DIM = 8, 6, 5, 3, 4
MEAN, STD = 0.5, 0.25
MEMBERS, STEPS = 6, 5


class HistoryMeanGenerator(nn.Module):
    """Return the mean of the history channels plus a z-driven offset: what it saw, readable."""

    def __init__(self) -> None:
        """Carry the two attributes the rollout reads off a real model."""
        super().__init__()
        self.hparams = type("H", (), {"z_dim": Z_DIM})()
        self.statistics = {"targets_mean": np.array(MEAN), "targets_std": np.array(STD)}

    def forward(
        self, inputs: torch.Tensor, timestamps: torch.Tensor, mask: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Mean over the history channels, shifted by the first z coordinate."""
        history = inputs[:, FEATURES:]
        return (history.mean(dim=1, keepdim=True) + 0.01 * z[:, :1, None, None]) * mask


def _rollout_inputs(
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    mask = torch.zeros(1, HEIGHT, WIDTH)
    mask[:, 1:5, 1:4] = 1.0
    feats = (
        torch.tensor(rng.standard_normal((STEPS, FEATURES, HEIGHT, WIDTH)), dtype=torch.float32)
        * mask
    )
    history = torch.tensor(rng.standard_normal((HISTORY, HEIGHT, WIDTH)), dtype=torch.float32)
    history = history * mask
    timestamps = torch.arange(STEPS)
    z = torch.tensor(rng.standard_normal((STEPS, MEMBERS, Z_DIM)), dtype=torch.float32)
    return feats, history, timestamps, mask, z


def _run(scale: torch.Tensor | None = None, shift: torch.Tensor | None = None) -> np.ndarray:
    feats, history, timestamps, mask, z = _rollout_inputs()
    return compute_trajectory_iterative(
        HistoryMeanGenerator(),
        feats,
        history,
        timestamps,
        mask,
        z,
        np.float32,
        history_scale=scale,
        history_shift=shift,
    )


def _region() -> np.ndarray:
    return _rollout_inputs()[3].numpy().astype(bool)[0]


def test_identity_perturbation_is_bit_exact_with_the_unperturbed_path() -> None:
    """Scale 1 and shift 0 take the perturbed path and must reproduce the plain one exactly."""
    plain = _run()
    identity = _run(scale=torch.ones(MEMBERS), shift=torch.zeros(MEMBERS))
    assert plain.shape == (MEMBERS, STEPS, 1, HEIGHT, WIDTH)
    assert np.array_equal(plain, identity)


def test_neutral_members_see_a_zero_history_and_keep_their_own_feedback() -> None:
    """Scale 0 shows the training mean at every step; the fed-back buffer is never rescaled."""
    scale = torch.ones(MEMBERS)
    scale[:2] = 0.0
    out = _run(scale=scale, shift=torch.zeros(MEMBERS))
    mask = _region()
    z = _rollout_inputs()[4].numpy()
    # Step 0: a neutral member sees history * 0, so its output is the training mean (+ the tiny
    # z term); a plain member reproduces the mean of the observed history.
    for member in range(2):
        expected = MEAN + STD * 0.01 * z[0, member, 0]
        assert np.allclose(out[member, 0, 0][mask], expected, atol=1e-5)
    history = _rollout_inputs()[1].numpy()
    plain_expected = MEAN + STD * (history.mean(axis=0) + 0.01 * z[0, 2, 0])
    assert np.allclose(out[2, 0, 0][mask], plain_expected[mask], atol=1e-5)
    # At step 1 the neutral member still sees zeros -- the scale applies to the input copy at
    # every step -- so its output stays at the mean.
    assert np.allclose(out[0, 1, 0][mask], MEAN + STD * 0.01 * z[1, 0, 0], atol=1e-5)
    # Padding never leaves zero.
    assert np.all(out[:, :, 0][:, :, ~mask] == 0.0)


def test_history_shift_moves_what_the_generator_sees_by_exactly_the_shift() -> None:
    """A per-member shift of the history moves the output by std * shift at the first step."""
    shift = torch.tensor([0.0, 0.7, -0.4, 0.0, 0.0, 0.0])
    out = _run(scale=torch.ones(MEMBERS), shift=shift)
    plain = _run()
    mask = _region()
    assert np.allclose(out[1, 0, 0][mask] - plain[1, 0, 0][mask], STD * 0.7, atol=1e-5)
    assert np.allclose(out[2, 0, 0][mask] - plain[2, 0, 0][mask], -STD * 0.4, atol=1e-5)
    assert np.array_equal(out[0], plain[0])


def test_rollout_rejects_wrong_sized_perturbations() -> None:
    """One value per trajectory, checked before anything is reshaped."""
    with pytest.raises(ValueError, match="one value per trajectory"):
        _run(scale=torch.ones(3))


def test_member_perturbations_are_deterministic_and_validated() -> None:
    """Same seed, same ensemble; both knobs off is the unperturbed signal."""
    assert member_perturbations(20, 0, 0.0) == (None, None)
    scale, shift = member_perturbations(20, 4, 1.0, seed=3)
    assert scale.tolist()[:4] == [0.0] * 4
    assert scale[4:].eq(1.0).all()
    assert torch.equal(shift, member_perturbations(20, 4, 1.0, seed=3)[1])
    assert not torch.equal(shift, member_perturbations(20, 4, 1.0, seed=4)[1])
    assert abs(float(shift.std()) - 1.0) < 0.4
    scale_only, shift_zero = member_perturbations(20, 4, 0.0)
    assert scale_only is not None
    assert shift_zero.abs().sum() == 0
    with pytest.raises(ValueError, match="neutral_members"):
        member_perturbations(20, 21, 0.0)
    with pytest.raises(ValueError, match="state_sd"):
        member_perturbations(20, 0, -1.0)


def test_seasonal_offsets_recover_a_known_monthly_bias_and_ignore_unseen_months() -> None:
    """Pooled over windows with different month coverage, the fitted offsets are the bias."""
    rng = np.random.default_rng(0)
    true_bias = np.linspace(-0.1, 0.1, 12)
    windows = []
    for start in (0, 5):  # two windows with different month coverage
        months = (np.arange(24) + start) % 12
        truth = rng.normal(0.5, 0.2, size=(24, 30))
        ens = truth + true_bias[months][:, None] + 0.001 * rng.standard_normal((24, 30))
        windows.append((ens, truth, months))
    offsets = fit_seasonal_offsets(*zip(*windows, strict=True))
    assert np.allclose(offsets, true_bias, atol=0.002)
    ens, truth, months = windows[0]
    partial = fit_seasonal_offsets([ens[:3]], [truth[:3]], [months[:3]])
    assert np.count_nonzero(partial) == 3
    assert np.allclose(partial[:3], true_bias[:3], atol=0.002)
    with pytest.raises(ValueError, match="align"):
        fit_seasonal_offsets([ens], [truth[:-1]], [months])


def test_apply_seasonal_offsets_on_cells_and_on_maps_keeps_the_padding_at_zero() -> None:
    """The offset of each step's month is subtracted on the region only."""
    offsets = np.arange(12) / 100.0
    months = np.array([0, 5, 11, 3])
    members = np.ones((3, 4, 7))
    out = apply_seasonal_offsets(members, months, offsets)
    assert np.allclose(out[:, :, 0], 1.0 - offsets[months][None])
    mask = np.zeros((1, 4, 3))
    mask[0, 1:3, :2] = 1.0
    maps = np.ones((3, 4, 1, 4, 3)) * mask
    out_maps = apply_seasonal_offsets(maps, months, offsets, mask)
    assert np.all(out_maps[:, :, 0][:, :, ~mask[0].astype(bool)] == 0.0)
    assert np.allclose(out_maps[:, 2, 0, 1, 0], 1.0 - 0.11)
    with pytest.raises(ValueError, match="12 monthly"):
        apply_seasonal_offsets(members, months, offsets[:5])
    with pytest.raises(ValueError, match="one month per step"):
        apply_seasonal_offsets(members, months[:2], offsets)


def test_month_index_reads_the_rasters_timestamp_encoding() -> None:
    """``dataframe_to_rasters`` stores ``month - 1`` as a float column."""
    stamps = np.array([[0.0], [11.0], [5.0]], dtype=np.float32)
    assert month_index(stamps).tolist() == [0, 11, 5]
    with pytest.raises(ValueError, match="0..11"):
        month_index(np.array([[12.0]]))


def test_spec_round_trips_through_json_and_is_validated(tmp_path: Path) -> None:
    """What the selection writes is what inference reads, with the fields it needs checked."""
    spec = dispersion_spec(4, 0.6, np.linspace(-0.05, 0.05, 12), member_seed=7, checkpoint="x.ckpt")
    path = tmp_path / "spec.json"
    save_dispersion_spec(spec, path)
    loaded = load_dispersion_spec(path)
    assert loaded["neutral_members"] == 4
    assert loaded["state_sd"] == 0.6
    assert loaded["member_seed"] == 7
    assert len(loaded["seasonal_offsets"]) == 12
    assert loaded["checkpoint"] == "x.ckpt"
    assert load_dispersion_spec({"neutral_members": 2, "state_sd": 0.0})["seasonal_offsets"] is None
    with pytest.raises(ValueError, match="missing"):
        load_dispersion_spec({"state_sd": 1.0})
    bad = {"neutral_members": 1, "state_sd": 1.0, "seasonal_offsets": [0.0] * 5}
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="12 values"):
        load_dispersion_spec(path)
