"""Tests of what writes the generator's BatchNorm running statistics, and of their recalibration.

Every eval-mode score -- the free-running rollout checkpoint selection ranks on, the
teacher-forced validation, inference -- normalizes by the generator's BatchNorm running
statistics. On the 500-epoch Bourgogne-Franche-Comte run those were a draw of the last training
batches of each epoch, worth +-0.05 of val rollout RMSE. These tests pin the fixes: the critic's
generator passes no longer write the statistics, the recalibration replaces them with an
average that depends on the weights and not on what was stored, it moves nothing else, and the
checkpoint Lightning saves after validation carries the recalibrated values.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from modules.batchnorm import batchnorm_layers, frozen_running_statistics
from modules.masking import MaskedBatchNorm2d
from swigan.engines.swigan_lit import TTTSWIGAN
from utils.batchnorm_recalibration import recalibrate_batchnorm
from utils.callbacks import BatchNormRecalibrationCallback
from utils.swi_dataset import SWIDataset

FEATURES, HISTORY, HEIGHT, WIDTH, Z_DIM = 2, 8, 23, 11, 4
STATISTICS = ("running_mean", "running_var", "num_batches_tracked")


def _model() -> TTTSWIGAN:
    model = TTTSWIGAN(
        input_channels=FEATURES + HISTORY,
        output_channels=1,
        input_map_dims=[HEIGHT, WIDTH],
        encoder_channels=[16, 16, 16, 16, 16],
        decoder_channels=[16, 16, 16, 16, 16],
        timestamps_dim=2,
        spatial_dropout=0.0,
        apply_center_block=True,
        z_dim=Z_DIM,
        lr=1e-3,
        weight_decay=0.0,
        loss_fn="l1",
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
        max_epochs=1,
    )
    model.input_statistics = {"targets_mean": np.array(0.5), "targets_std": np.array(0.25)}
    return model


def _dataset(months: int = HISTORY + 20) -> SWIDataset:
    rng = np.random.default_rng(0)
    mask = np.zeros((1, HEIGHT, WIDTH), dtype=np.float32)
    mask[:, 2:21, 3:9] = 1.0
    feats = rng.standard_normal((months, FEATURES, HEIGHT, WIDTH)).astype(np.float32)
    targets = rng.standard_normal((months, 1, HEIGHT, WIDTH)).astype(np.float32)
    stamps = (np.arange(months) % 12).astype(np.float32).reshape(-1, 1)
    return SWIDataset(feats, targets, mask, stamps, HISTORY)


def _batch(dataset: SWIDataset, size: int) -> dict[str, torch.Tensor]:
    return next(iter(DataLoader(dataset, batch_size=size, shuffle=False)))


def _running_statistics(model: TTTSWIGAN) -> dict[str, torch.Tensor]:
    return {
        name: buffer.clone()
        for name, buffer in model.generator.named_buffers()
        if name.rsplit(".", 1)[-1] in STATISTICS
    }


def _assert_statistics_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys()
    for name in a:
        assert torch.equal(a[name], b[name]), name


def test_masked_batch_norm_without_tracking_normalizes_by_the_batch_and_writes_nothing() -> None:
    """``track_running_stats=False`` is the switch ``nn.BatchNorm2d`` reads; the masked one too."""
    tracked = MaskedBatchNorm2d(3).train()
    frozen = copy.deepcopy(tracked)
    frozen.track_running_stats = False
    values, mask = torch.randn(4, 3, 8, 8), torch.zeros(4, 1, 8, 8)
    mask[..., 2:6, 1:7] = 1.0

    torch.testing.assert_close(frozen(values, mask), tracked(values, mask))
    assert torch.equal(frozen.running_mean, torch.zeros(3))
    assert torch.equal(frozen.running_var, torch.ones(3))
    assert int(frozen.num_batches_tracked) == 0
    assert int(tracked.num_batches_tracked) == 1


def test_frozen_running_statistics_leaves_the_forward_pass_unchanged() -> None:
    """Same output, batch statistics and all; only the buffers are spared."""
    model = _model().train()
    batch = _batch(_dataset(), 6)
    before = _running_statistics(model)

    def forward() -> torch.Tensor:
        torch.manual_seed(1)
        return model(batch["input_maps"], batch["timestamps"], batch["mask"])

    with frozen_running_statistics(model.generator):
        frozen = forward()
    _assert_statistics_equal(_running_statistics(model), before)
    assert all(layer.track_running_stats for layer in batchnorm_layers(model.generator))

    torch.testing.assert_close(frozen, forward())
    assert not torch.equal(
        _running_statistics(model)["encoder.layers.0.conv1.1.running_mean"],
        before["encoder.layers.0.conv1.1.running_mean"],
    )


def test_critic_step_leaves_the_running_statistics_to_the_generator_step() -> None:
    """Five critic passes on the same batch used to write it five more times than training did."""
    model = _model().train()
    batch = _batch(_dataset(), 6)
    z = torch.randn(6, Z_DIM)
    before = _running_statistics(model)

    model.critic_step(batch, z)
    _assert_statistics_equal(_running_statistics(model), before)

    model.generator_step(batch, z)
    counts = [int(layer.num_batches_tracked) for layer in batchnorm_layers(model.generator)]
    assert counts
    assert set(counts) == {1}


def test_recalibration_does_not_depend_on_the_stored_statistics() -> None:
    """Whatever the last batches left behind, the same weights recalibrate to the same values."""
    torch.manual_seed(0)
    model = _model()
    scrambled = copy.deepcopy(model)
    for layer in batchnorm_layers(scrambled.generator):
        layer.running_mean.normal_()
        layer.running_var.uniform_(0.5, 3.0)
        layer.num_batches_tracked.fill_(7)
    dataset = _dataset()

    assert recalibrate_batchnorm(model, dataset, batch_size=8, seed=3) > 0
    recalibrate_batchnorm(scrambled, dataset, batch_size=8, seed=3)
    _assert_statistics_equal(_running_statistics(model), _running_statistics(scrambled))

    # A cumulative average over every batch of every pass, 20 samples in batches of 8.
    recalibrate_batchnorm(model, dataset, batch_size=8, num_passes=2, seed=3)
    counts = {int(layer.num_batches_tracked) for layer in batchnorm_layers(model.generator)}
    assert counts == {2 * math.ceil(len(dataset) / 8)}


def test_recalibration_moves_nothing_but_the_running_statistics() -> None:
    """Weights, module modes, momenta and the global RNG are exactly as the caller left them."""
    model = _model().eval()
    model.generator.encoder.layers[0].train()  # a mixed state, as inference's dropout mode is
    modes = [module.training for module in model.modules()]
    weights = {name: param.clone() for name, param in model.named_parameters()}
    momenta = [layer.momentum for layer in batchnorm_layers(model.generator)]
    rng = torch.get_rng_state()

    recalibrate_batchnorm(model, _dataset(), batch_size=8)

    assert [module.training for module in model.modules()] == modes
    for name, param in model.named_parameters():
        assert torch.equal(param, weights[name]), name
    assert [layer.momentum for layer in batchnorm_layers(model.generator)] == momenta
    assert torch.equal(torch.get_rng_state(), rng)


def test_the_checkpoint_saved_after_validation_carries_the_recalibrated_statistics(
    tmp_path: Path,
) -> None:
    """What ModelCheckpoint writes is the recalibration of the weights it writes, not the EMA."""
    torch.manual_seed(0)
    model = _model()
    dataset = _dataset()
    trainer = Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            BatchNormRecalibrationCallback(dataset, batch_size=8, seed=3),
            ModelCheckpoint(dirpath=tmp_path, save_last=True),
        ],
    )
    trainer.fit(
        model,
        DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True),
        DataLoader(dataset, batch_size=8),
    )
    saved = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=False)
    saved_statistics = {
        name.removeprefix("generator."): value
        for name, value in saved["state_dict"].items()
        if name.startswith("generator.") and name.rsplit(".", 1)[-1] in STATISTICS
    }

    reference = _model()
    reference.load_state_dict(saved["state_dict"])
    recalibrate_batchnorm(reference, dataset, batch_size=8, seed=3)
    _assert_statistics_equal(saved_statistics, _running_statistics(reference))
    # Two training batches would have left a count of 2; the recalibration's three batches, 3.
    assert {int(v) for k, v in saved_statistics.items() if k.endswith("num_batches_tracked")} == {
        math.ceil(len(dataset) / 8)
    }
