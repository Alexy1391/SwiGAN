"""Tests of the mini-ensemble pixel term and of the history-dropout augmentation.

The fair CRPS and the energy score are the only terms in the generator objective that credit
spread, so what they must get right is (1) the estimator -- the fair form of Ferro et al., which
the offline scorer already uses -- and (2) properness: against a truth drawn from a known
distribution, an ensemble at the true spread scores better than a collapsed one and than an
over-dispersed one. The single-sample path has to be the L1 term it always was.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from swigan.engines.swigan_lit import TTTSWIGAN
from utils.ensemble_scores import crps_fair
from utils.swi_dataset import SWIDataset, build_datasets_from_split_rasters

FEATURES, HISTORY, HEIGHT, WIDTH, Z_DIM = 2, 8, 23, 11, 4


def _model(loss_fn: str, ensemble_size: int) -> TTTSWIGAN:
    model = TTTSWIGAN(
        input_channels=FEATURES + HISTORY,
        output_channels=1,
        input_map_dims=[HEIGHT, WIDTH],
        encoder_channels=[32, 32, 32, 32, 32],
        decoder_channels=[32, 32, 32, 32, 32],
        timestamps_dim=2,
        spatial_dropout=0.0,
        apply_center_block=True,
        z_dim=Z_DIM,
        lr=1e-3,
        weight_decay=0.0,
        loss_fn=loss_fn,
        optim=torch.optim.AdamW,
        normalization="batchnorm",
        patch_critic_loss="wasserstein",
        patch_aggregation="learned",
        gradient_penalty_reduction="mean",
        lambda_penalty=1.0,
        image_distance_weight=12.0,
        feature_matching_weight=1.2,
        critic_on_padded_canvas=True,
        diff_augment_on_region=True,
        ensemble_size=ensemble_size,
    )
    model.input_statistics = {"targets_mean": np.array(0.5), "targets_std": np.array(0.25)}
    return model


def _mask(batch: int) -> torch.Tensor:
    mask = torch.zeros(batch, 1, HEIGHT, WIDTH)
    mask[:, :, 2:21, 3:9] = 1.0
    return mask


def _reference_crps(samples: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Apply the offline estimator per sample over its region cells, averaged over the batch."""
    per_sample = []
    for b in range(target.shape[0]):
        cells = mask[b, 0].astype(bool)
        per_sample.append(crps_fair(samples[:, b, 0][:, cells], target[b, 0][cells]))
    return float(np.mean(per_sample))


def _batch(batch: int) -> dict[str, torch.Tensor]:
    mask = _mask(batch)
    return {
        "input_maps": torch.randn(batch, FEATURES + HISTORY, HEIGHT, WIDTH) * mask,
        "target_maps": torch.randn(batch, 1, HEIGHT, WIDTH) * mask,
        "timestamps": torch.randint(0, 12, (batch,)),
        "mask": mask,
    }


def _rasters(months: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    mask = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    mask[:, 2:21, 3:9] = 1.0
    feats = rng.standard_normal((months, FEATURES, HEIGHT, WIDTH)).astype(np.float32)
    targets = rng.standard_normal((months, 1, HEIGHT, WIDTH)).astype(np.float32)
    stamps = (np.arange(months) % 12).astype(np.float32).reshape(-1, 1)
    return feats, targets, stamps, mask


def test_fair_crps_matches_the_offline_estimator() -> None:
    """The per-cell fair CRPS in the loss is ``utils.ensemble_scores.crps_fair`` on the region."""
    torch.manual_seed(0)
    model = _model("crps", 5)
    k, batch = 5, 3
    mask = _mask(batch)
    target = torch.randn(batch, 1, HEIGHT, WIDTH) * mask
    samples = (target.unsqueeze(0) + 0.3 * torch.randn(k, batch, 1, HEIGHT, WIDTH)) * mask
    score, spread = model.ensemble_pixel_distance(samples, target, mask)
    reference = _reference_crps(samples.numpy(), target.numpy(), mask.numpy())
    assert score.item() == pytest.approx(reference, rel=1e-5)
    assert spread.item() > 0


def test_energy_score_with_two_samples_is_the_hand_formula() -> None:
    """With K = 2 the pair weight 1/(K(K-1)) is 1/2, on the rms distance over the region."""
    torch.manual_seed(1)
    model = _model("energy", 2)
    batch = 2
    mask = _mask(batch)
    target = torch.randn(batch, 1, HEIGHT, WIDTH) * mask
    samples = (target.unsqueeze(0) + 0.3 * torch.randn(2, batch, 1, HEIGHT, WIDTH)) * mask
    score, spread = model.ensemble_pixel_distance(samples, target, mask)

    cells = mask[0, 0].bool()

    def rms(diff: torch.Tensor) -> float:
        return float(torch.sqrt((diff[0][cells] ** 2).mean()))

    expected, pairs = [], []
    for b in range(batch):
        skill = 0.5 * (rms(samples[0, b] - target[b]) + rms(samples[1, b] - target[b]))
        pair = rms(samples[0, b] - samples[1, b])
        expected.append(skill - 0.5 * pair)
        pairs.append(0.5 * pair)
    assert score.item() == pytest.approx(float(np.mean(expected)), rel=1e-4)
    assert spread.item() == pytest.approx(float(np.mean(pairs)), rel=1e-4)


@pytest.mark.parametrize("kind", ["crps", "energy"])
def test_proper_scores_are_minimised_at_the_true_spread(kind: str) -> None:
    """Against a truth with sd 0.2, an ensemble at sd 0.2 beats a collapsed and a 3x-wide one."""
    torch.manual_seed(2)
    model = _model(kind, 8)
    batch, true_sd = 64, 0.2
    mask = _mask(batch)
    mu = torch.randn(batch, 1, HEIGHT, WIDTH)
    # Coherent over the region, so the energy score has something to credit.
    target = (mu + true_sd * torch.randn(batch, 1, 1, 1)) * mask
    scores = {}
    for sd in (0.0, true_sd, 3 * true_sd):
        samples = (mu.unsqueeze(0) + sd * torch.randn(8, batch, 1, 1, 1)) * mask
        scores[sd] = model.ensemble_pixel_distance(samples, target, mask)[0].item()
    assert scores[true_sd] < scores[0.0]
    assert scores[true_sd] < scores[3 * true_sd]


def test_energy_prefers_coherent_spread_where_crps_cannot_tell() -> None:
    """Coherent spread scores better than pixel speckle of the same per-cell size.

    That is the point of choosing the energy score: the per-cell CRPS gives the two the same
    mark because their marginals are the same.
    """
    torch.manual_seed(3)
    # A coherent ensemble has one independent realisation per MAP, not per cell, so the
    # comparison needs many maps before its sampling noise is below the effect.
    batch, sd = 512, 0.2
    mask = _mask(batch)
    mu = torch.randn(batch, 1, HEIGHT, WIDTH)
    target = (mu + sd * torch.randn(batch, 1, 1, 1)) * mask  # coherent truth
    collapsed = (mu.unsqueeze(0).expand(8, -1, -1, -1, -1)) * mask
    speckle = (mu.unsqueeze(0) + sd * torch.randn(8, batch, 1, HEIGHT, WIDTH)) * mask
    coherent = (mu.unsqueeze(0) + sd * torch.randn(8, batch, 1, 1, 1)) * mask
    arms = (("collapsed", collapsed), ("speckle", speckle), ("coherent", coherent))
    energy = _model("energy", 8)
    crps = _model("crps", 8)
    e = {n: energy.ensemble_pixel_distance(s, target, mask)[0].item() for n, s in arms}
    c = {n: crps.ensemble_pixel_distance(s, target, mask)[0].item() for n, s in arms}
    assert e["coherent"] < e["speckle"] < e["collapsed"]
    assert c["speckle"] == pytest.approx(c["coherent"], rel=0.05)  # same marginals, same CRPS
    assert c["speckle"] < c["collapsed"]


def test_single_sample_paths_are_the_l1_term() -> None:
    """One sample through a proper score reads as L1, and the size checks fire."""
    torch.manual_seed(4)
    batch = 3
    mask = _mask(batch)
    target = torch.randn(batch, 1, HEIGHT, WIDTH) * mask
    fake = torch.randn(batch, 1, HEIGHT, WIDTH) * mask
    l1 = _model("l1", 1).pixel_distance(fake, target, mask)
    for kind in ("crps", "energy"):
        single = _model(kind, 2).pixel_distance(fake, target, mask)
        assert single.item() == pytest.approx(l1.item())
    with pytest.raises(ValueError, match="ensemble_size >= 2"):
        _model("energy", 1)
    with pytest.raises(ValueError, match="at least 1"):
        _model("l1", 0)
    with pytest.raises(ValueError, match="at least 2 samples"):
        _model("energy", 2).ensemble_pixel_distance(fake.unsqueeze(0), target, mask)


def test_generate_samples_shapes_and_the_single_sample_identity() -> None:
    """K = 1 is ``forward`` unchanged; K = 4 stacks four distinct samples, zero off the region."""
    torch.manual_seed(5)
    batch = 2
    data = _batch(batch)
    z = torch.randn(batch, Z_DIM)
    one = _model("l1", 1)
    one.eval()
    torch.manual_seed(6)
    direct = one(data["input_maps"], data["timestamps"], data["mask"], noise_vector=z)
    torch.manual_seed(6)
    stacked = one.generate_samples(data["input_maps"], data["timestamps"], data["mask"], z)
    assert stacked.shape == (1, batch, 1, HEIGHT, WIDTH)
    assert torch.equal(stacked[0], direct)

    four = _model("energy", 4)
    four.eval()
    samples = four.generate_samples(data["input_maps"], data["timestamps"], data["mask"], z)
    assert samples.shape == (4, batch, 1, HEIGHT, WIDTH)
    assert not torch.equal(samples[0], samples[1])  # different z and injection draws
    assert torch.all(samples[:, :, :, ~data["mask"][0, 0].bool()] == 0)


def test_generator_step_scores_the_mini_ensemble_and_backpropagates() -> None:
    """The generator step returns a finite loss with a positive spread term that has a gradient."""
    torch.manual_seed(7)
    batch = 2
    model = _model("energy", 3)
    model.train()
    data = _batch(batch)
    out = model.generator_step(data, torch.randn(batch, Z_DIM))
    assert torch.isfinite(out["loss_generator"])
    assert out["pixel_spread"].item() > 0
    assert out["pixel_distance_loss"].shape == ()
    out["loss_generator"].backward()
    grads = [p.grad for p in model.generator.parameters() if p.grad is not None]
    assert grads
    assert any(g.abs().sum() > 0 for g in grads)

    plain = _model("l1", 1)
    plain.train()
    plain_out = plain.generator_step(data, torch.randn(batch, Z_DIM))
    assert plain_out["pixel_spread"].item() == 0.0


def test_history_dropout_scales_only_the_history_frames_of_training_samples() -> None:
    """The factor hits the history channels of a served sample, never the stored one or val."""
    feats, targets, stamps, mask = _rasters()
    plain = SWIDataset(feats, targets, mask, stamps, HISTORY)
    always = SWIDataset(
        feats,
        targets,
        mask,
        stamps,
        HISTORY,
        history_dropout_prob=1.0,
        history_dropout_range=(0.5, 0.5),
    )
    reference, dropped = plain[3]["input_maps"], always[3]["input_maps"]
    assert torch.equal(dropped[:FEATURES], reference[:FEATURES])
    assert torch.allclose(dropped[FEATURES:], 0.5 * reference[FEATURES:])
    assert torch.equal(plain[3]["input_maps"], reference)  # the stored sample is untouched
    assert torch.equal(always[3]["target_maps"], plain[3]["target_maps"])

    never = SWIDataset(feats, targets, mask, stamps, HISTORY, history_dropout_prob=0.0)
    assert torch.equal(never[3]["input_maps"], reference)
    with pytest.raises(ValueError, match="history_dropout_range"):
        SWIDataset(
            feats,
            targets,
            mask,
            stamps,
            HISTORY,
            history_dropout_prob=0.5,
            history_dropout_range=(0.8, 0.2),
        )
    with pytest.raises(ValueError, match="history_dropout_prob"):
        SWIDataset(feats, targets, mask, stamps, HISTORY, history_dropout_prob=1.5)

    datasets, _ = build_datasets_from_split_rasters(
        split_maps={"train": feats, "val": feats[:12]},
        split_targets={"train": targets, "val": targets[:12]},
        split_timesteps={"train": stamps, "val": stamps[:12]},
        mask=mask,
        num_input_steps=HISTORY,
        history_dropout_prob=1.0,
        history_dropout_range=(0.0, 0.0),
    )
    assert datasets["train"].history_dropout_prob == 1.0
    assert datasets["val"].history_dropout_prob == 0.0
    assert torch.all(datasets["train"][0]["input_maps"][FEATURES:] == 0)
