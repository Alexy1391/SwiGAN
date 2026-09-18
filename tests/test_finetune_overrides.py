"""Tests of fine-tuning from a checkpoint with this config's objective and schedule.

``swigan/train.py`` loads the weights and overrides the loss, the learning-rate schedule,
the ensemble size and the noise-gain learning rate from the current config, because the saved
hyper-parameters would otherwise rebuild the old run's cosine schedule and its param groups.
This pins that the overrides land, that ``noise_weight_lr_scale`` reaches the optimizer rather
than only the hparams, that a loaded module carries no ``input_statistics`` until the trainer
sets them, and that the weights are the checkpoint's.
"""

from __future__ import annotations

from pathlib import Path

import lightning
import numpy as np
import torch

from swigan.engines.swigan_lit import TTTSWIGAN

FEATURES, HISTORY, HEIGHT, WIDTH = 2, 8, 23, 11


def _model(
    loss_fn: str = "l1",
    ensemble_size: int = 1,
    max_epochs: int = 250,
    noise_weight_lr_scale: float = 1.0,
) -> TTTSWIGAN:
    return TTTSWIGAN(
        input_channels=FEATURES + HISTORY,
        output_channels=1,
        input_map_dims=[HEIGHT, WIDTH],
        encoder_channels=[32, 32, 32, 32, 32],
        decoder_channels=[32, 32, 32, 32, 32],
        timestamps_dim=2,
        spatial_dropout=0.0,
        apply_center_block=True,
        z_dim=4,
        lr=1e-5,
        weight_decay=0.0,
        loss_fn=loss_fn,
        optim=torch.optim.AdamW,
        normalization="batchnorm",
        max_epochs=max_epochs,
        min_lr=1e-7,
        ensemble_size=ensemble_size,
        noise_weight_lr_scale=noise_weight_lr_scale,
    )


def _save_checkpoint(model: TTTSWIGAN, path: Path) -> None:
    """Write the three keys ``load_from_checkpoint`` reads, the way Lightning lays them out."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        path,
    )


def test_overrides_replace_the_saved_objective_and_schedule(tmp_path: Path) -> None:
    """A 250-epoch L1 checkpoint becomes a 50-epoch energy fine-tune with K = 4 samples."""
    torch.manual_seed(0)
    source = _model()
    path = tmp_path / "last.ckpt"
    _save_checkpoint(source, path)

    overrides = {
        "loss_fn": "energy",
        "lr": 2e-5,
        "min_lr": 1e-8,
        "max_epochs": 50,
        "ensemble_size": 4,
        "image_distance_weight": 12.0,
        "feature_matching_weight": 1.2,
        "noise_weight_lr_scale": 100.0,
    }
    loaded = TTTSWIGAN.load_from_checkpoint(path, map_location="cpu", **overrides)
    assert loaded.hparams.max_epochs == 50
    assert loaded.hparams.lr == 2e-5
    assert loaded.hparams.min_lr == 1e-8
    assert loaded.hparams.ensemble_size == 4
    assert loaded.hparams.noise_weight_lr_scale == 100.0
    assert loaded.pixel_distance_kind == "energy"
    # What the trainer then sets; a loaded module has none of its own.
    assert getattr(loaded, "input_statistics", None) is None
    loaded.input_statistics = {"targets_mean": np.array(0.5), "targets_std": np.array(0.25)}
    # The weights are the checkpoint's, not a fresh draw.
    for (name, before), (_, after) in zip(
        source.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert torch.equal(before, after), name
    # The schedule is rebuilt over the fine-tune's horizon.
    _, schedulers = loaded.configure_optimizers()
    assert schedulers[0].T_max == 50
    assert schedulers[0].eta_min == 1e-8


def test_an_old_checkpoint_without_the_new_field_loads_with_one_sample(tmp_path: Path) -> None:
    """Checkpoints written before ``ensemble_size`` existed default to the single-sample term."""
    source = _model()
    hparams = dict(source.hparams)
    del hparams["ensemble_size"]
    path = tmp_path / "old.ckpt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "hyper_parameters": hparams,
            "pytorch-lightning_version": lightning.__version__,
        },
        path,
    )
    loaded = TTTSWIGAN.load_from_checkpoint(path, map_location="cpu", loss_fn="l1")
    assert loaded.hparams.ensemble_size == 1
    assert loaded.pixel_distance_kind == "l1"


def test_the_noise_gain_rate_overrides_the_checkpoints_and_reaches_the_optimizer(
    tmp_path: Path,
) -> None:
    """The scale is read in ``configure_optimizers``, so the hparam alone proves nothing.

    A module loaded from a checkpoint rebuilds its optimizers from the SAVED hparams. The
    250-epoch base run trained at scale 1.0, at which the per-channel gains move ~1e-3 over the
    whole schedule -- the injection is then a fixed random draw, not something a fine-tune can
    move. This pins that the fine-tune's scale replaces the checkpoint's AND lands as the
    learning rate of the group that actually holds the ``noise_weights`` parameters.
    """
    source = _model(noise_weight_lr_scale=1.0)
    path = tmp_path / "base.ckpt"
    _save_checkpoint(source, path)

    # ``loss_fn`` is not among the saved hyper-parameters, so every load must supply it.
    loaded = TTTSWIGAN.load_from_checkpoint(
        path,
        map_location="cpu",
        loss_fn="energy",
        ensemble_size=4,
        lr=3e-6,
        noise_weight_lr_scale=100.0,
    )
    assert loaded.hparams.noise_weight_lr_scale == 100.0

    (optimizer_generator, _), _ = loaded.configure_optimizers()
    groups = optimizer_generator.param_groups
    assert len(groups) == 2, "the gains need their own group or the scale cannot apply"
    conv_group, noise_group = groups
    assert noise_group["lr"] == 3e-6 * 100.0
    assert conv_group["lr"] == 3e-6

    # The split is by name, so check the group holds every gain and nothing else.
    gains = {id(p) for name, p in loaded.generator.named_parameters() if "noise_weights" in name}
    assert gains, "the generator has no noise gains to scale"
    assert {id(p) for p in noise_group["params"]} == gains
    assert not ({id(p) for p in conv_group["params"]} & gains)


def test_without_the_override_the_checkpoints_scale_survives(tmp_path: Path) -> None:
    """The failure this guards against: a run launched at scale 100 that trains at scale 1.0."""
    source = _model(noise_weight_lr_scale=1.0)
    path = tmp_path / "base.ckpt"
    _save_checkpoint(source, path)

    loaded = TTTSWIGAN.load_from_checkpoint(
        path, map_location="cpu", loss_fn="energy", ensemble_size=4, lr=3e-6
    )
    (optimizer_generator, _), _ = loaded.configure_optimizers()
    assert optimizer_generator.param_groups[1]["lr"] == 3e-6
