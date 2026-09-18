"""Free-running multi-step rollout of the generator, scored the way the paper scores it.

WHY THIS EXISTS. ``val/generator_rmse`` is TEACHER-FORCED: every sample is one month predicted
from eight *observed* SWI frames, so the error never compounds and the metric cannot see a
generator that has learned to copy its most recent input. The number the paper reports in
Section 5.1 is FREE-RUNNING: the model is seeded once with observed history and then runs the
whole horizon on its own predictions, each month's output becoming the next month's input.

The two are not the same quantity and they do not have the same minimum, so a checkpoint
selected on the teacher-forced curve is not the checkpoint that rolls out best. That gap is why
``run_paper1500_metrics.sh`` had to exist at all: it rescored finished runs offline because
nothing inside training measured the free-running error. This module puts that measurement on
the training curve, where it can drive checkpoint selection.

The horizon is the length of the split. With the chronological layout of
``config/train_swigan.yaml`` -- train 1960-2021, val 2022-2023, test 2024-2025, each seeded with
the last ``input_steps`` months of the split before it -- both val and test are exactly 24
samples, so ``val/rollout_rmse`` is a 24-step rollout RMSE over Jan 2022 - Dec 2023.

UNITS. The rollout feeds the generator's own output back into the SWI history channels, so
everything stays in the standardized space the model was trained in; the un-standardization to
physical SWI happens once, at the end, on the finished trajectory. (``swigan/inference.py`` seeds
its history from *raw* target maps instead, which mixes units for the first ``input_steps`` of
its rollout. That path is untouched here -- this module builds its seed from the dataset, which
is already standardized, so it does not inherit the problem.)

The metrics are the ones ``swigan/evaluate.py`` computes: per-pixel over the time axis, averaged
over ensemble members, then reduced over pixels. ``rmse_max`` is the paper's headline (0.21).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from lightning import LightningModule

from utils.ensemble_scores import crps_fair
from utils.metrics import (
    compute_rmse_per_date,
    compute_rmse_per_pixel,
    compute_smape_per_pixel,
)
from utils.spread_shape import BAND, MonthlySpreadAccumulator, spread_shape_verdict
from utils.swi_dataset import SWIDataset


@contextmanager
def _frozen_rng(seed: int) -> Iterator[None]:
    """Run the block on a private RNG stream, leaving the global one exactly as it was.

    The generator draws its per-block injection noise from the global ``torch.randn`` inside the
    forward pass (``unet_frame_encoder.py:85``, ``unet_frame_decoder.py:109``), so scoring a
    rollout mid-training consumes draws from the same stream the training steps use. Without
    this, every training draw after the first validation epoch would be shifted by however many
    numbers the rollout happened to consume, and the metric would be perturbing the run it is
    supposed to be measuring. (This restores the CPU and CUDA generator states, which is all it
    claims; cuDNN's non-deterministic kernels mean two identical runs still do not reproduce
    each other bit for bit on GPU.)
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _single_channel_statistic(statistics: dict[str, np.ndarray], name: str) -> float:
    """Read one standardization statistic as a scalar.

    They are saved with ``keepdims=True``, i.e. shaped ``(1, 1, 1, 1)``. The target has a single
    channel, so the value is a scalar; using it at its saved shape would broadcast the
    ``(T, H, W)`` observations up to 4-D and silently change what every metric below reduces
    over.
    """
    value = np.asarray(statistics[name], dtype=np.float32).reshape(-1)
    if value.size != 1:
        raise ValueError(
            f"'{name}' holds {value.size} values, so the target is not single-channel and the "
            "rollout's un-standardization would be wrong. This module assumes output_channels=1."
        )
    return float(value[0])


@contextmanager
def _eval_mode(model: torch.nn.Module) -> Iterator[None]:
    """Force eval mode for the rollout, then restore whatever mode the caller was in.

    Same requirement as ``swigan/inference.py:154``: in train mode stochastic depth drops
    residual blocks at random and BatchNorm normalizes by the statistics of the batch of
    fed-back predictions, which at 24 recursive steps is not the model the checkpoint describes.
    """
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def free_running_rollout(
    model: LightningModule,
    dataset: SWIDataset,
    statistics: dict[str, np.ndarray],
    num_members: int = 8,
    noise_std: float = 1.0,
    num_steps: int | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the generator forward over ``dataset`` without ever showing it an observed target.

    Sample ``i`` of a :class:`~utils.swi_dataset.SWIDataset` stacks the covariate maps of month
    ``i`` on top of the ``num_input_steps`` observed SWI frames that precede it. The rollout
    keeps the covariates -- they are forcing, not prediction -- but replaces that SWI history
    with a rolling buffer of the model's own outputs, seeded once from sample 0.

    Args:
    ----
        model: The trained ``TTTSWIGAN``.
        dataset: The split to roll over. Its samples must be in chronological order, which is
            how ``SWIDataset`` builds them.
        statistics: The training-split standardization statistics, i.e. the dict saved as
            ``train_dataset_statistics.npz``. Only ``targets_mean`` / ``targets_std`` are used.
        num_members: Ensemble members, run as one batch. They share the covariates and the seed
            history and differ only in their noise draws.
        noise_std: Scale of the bottleneck vector ``z``. 1.0 is what inference ships with.
        num_steps: Rollout horizon. ``None`` uses the whole split (24 months for the val and
            test windows of the chronological layout).
        seed: Seed of the private RNG stream the rollout runs on.
        device: Where to run. Defaults to the device the model is already on.

    Returns:
    -------
        ``(trajectories, y_true, mask)`` where ``trajectories`` is
        ``(num_members, num_steps, H, W)`` and ``y_true`` is ``(num_steps, H, W)``, both in
        physical SWI units and zeroed outside the region, and ``mask`` is a boolean ``(H, W)``.

    """
    horizon = len(dataset) if num_steps is None else int(num_steps)
    if horizon < 1:
        raise ValueError(f"num_steps must be at least 1, got {horizon}.")
    if horizon > len(dataset):
        raise ValueError(
            f"num_steps={horizon} exceeds the {len(dataset)} samples of this split. The split "
            "is shorter than the requested horizon -- either lower num_steps or check "
            "'dataset.carry_context_across_splits', which is what buys the split its first "
            "input_steps months back."
        )
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    num_history_steps = dataset.num_input_steps
    first = dataset[0]
    num_feature_channels = first["input_maps"].shape[0] - num_history_steps

    # Covariates, timestamps and observations for the whole horizon, stacked once.
    features = torch.stack(
        [dataset[i]["input_maps"][:num_feature_channels] for i in range(horizon)]
    ).to(device)
    timestamps = (
        torch.stack([dataset[i]["timestamps"].reshape(()) for i in range(horizon)])
        .to(device)
        .long()
    )
    observed = torch.stack([dataset[i]["target_maps"] for i in range(horizon)])  # (T, 1, H, W)
    mask = first["mask"].to(device)  # (1, H, W)

    # The one and only observed SWI the rollout ever sees.
    history = (
        first["input_maps"][num_feature_channels:]
        .to(device)
        .unsqueeze(0)
        .repeat(num_members, 1, 1, 1)
    )

    outputs = []
    with _eval_mode(model), _frozen_rng(seed), torch.no_grad():
        for step in range(horizon):
            inputs = torch.cat(
                [features[step].unsqueeze(0).expand(num_members, -1, -1, -1), history], dim=1
            )
            # One independent draw per step, as in Section 2.2: the Z_k are i.i.d. across k.
            z = torch.randn(num_members, model.hparams.z_dim, device=device) * noise_std
            prediction = model(
                inputs,
                timestamps[step].expand(num_members),
                mask.unsqueeze(0),
                z,
            )  # (M, 1, H, W)
            history = torch.cat([history, prediction], dim=1)[:, 1:]
            outputs.append(prediction)

    trajectories = torch.cat(outputs, dim=1)  # (M, T, H, W)

    # Scalars rather than the (1, 1, 1, 1) arrays as saved: the target has a single channel, and
    # a keepdims-shaped factor would broadcast the (T, H, W) observations up to 4-D.
    targets_mean = _single_channel_statistic(statistics, "targets_mean")
    targets_std = _single_channel_statistic(statistics, "targets_std")

    trajectories = (trajectories * targets_std + targets_mean) * mask
    y_true = (observed.squeeze(1).to(device) * targets_std + targets_mean) * mask

    return (
        trajectories.float().cpu().numpy(),
        y_true.float().cpu().numpy(),
        mask.squeeze(0).bool().cpu().numpy(),
    )


def summarise_rollout(
    trajectories: np.ndarray,
    y_true: np.ndarray,
    mask: np.ndarray,
    months: np.ndarray | None = None,
    target_std: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Score a finished rollout with the metrics of ``swigan/evaluate.py``.

    Each metric is computed per pixel over the time axis for every member, averaged over
    members, and only then reduced over pixels -- the paper's order, and the one the offline
    scorer already uses, so the numbers logged during training are on the same scale as the
    ones in ``Outputs/paper_metrics``.

    With two or more members the ensemble is also scored as a distribution: the fair CRPS
    and the spread-skill ratio of :mod:`utils.ensemble_scores`, and the per-calendar-month
    spread-shape test of :mod:`utils.spread_shape`, which is what a dispersion objective
    has to be held to -- an ensemble can sit at an annual spread-skill of 1.0 while being
    twice too wide in August and too narrow in November. ``months`` is needed for the
    monthly part; without it only the annual scores are added.

    Args:
    ----
        trajectories: ``(num_members, T, H, W)`` in physical SWI units.
        y_true: ``(T, H, W)`` observations in the same units.
        mask: Boolean ``(H, W)`` region of interest.
        months: 0-based calendar month of each step, ``(T,)``. Optional.
        target_std: ``targets_std`` of the training split. When given, ``rollout_crps`` is
            in the standardised units every offline CRPS in the project is quoted in;
            otherwise it is in SWI units.

    Returns:
    -------
        Scalar metrics, plus the per-date spatial RMSE curve under ``rmse_by_step`` and, when
        ``months`` is given, the twelve monthly ratios under ``spread_ratio_by_month``.

    """
    num_members = trajectories.shape[0]

    rmse_map = np.nanmean(
        np.stack(
            [compute_rmse_per_pixel(y_true, trajectories[m], mask) for m in range(num_members)]
        ),
        axis=0,
    )
    smape_map = np.nanmean(
        np.stack(
            [compute_smape_per_pixel(y_true, trajectories[m], mask) for m in range(num_members)]
        ),
        axis=0,
    )
    # Spatial RMSE per month, so the compounding the metric exists to expose is readable as a
    # curve rather than as a single pooled number.
    rmse_by_step = np.stack(
        [compute_rmse_per_date(y_true, trajectories[m], mask) for m in range(num_members)]
    ).mean(axis=0)

    metrics: dict[str, float | np.ndarray] = {}
    if num_members >= 2:
        members = trajectories[:, :, mask]  # (M, T, P), region cells only
        truth = y_true[:, mask]
        ens_mean = members.mean(axis=0)
        mse = float(((ens_mean - truth) ** 2).mean())
        mean_var = float(members.var(axis=0, ddof=1).mean())
        scale = float(target_std) if target_std else 1.0
        metrics["rollout_crps"] = crps_fair(members, truth) / scale
        metrics["rollout_spread_skill"] = (
            float(np.sqrt((num_members + 1) / num_members * mean_var / mse)) if mse > 0 else 0.0
        )
        if months is not None:
            # Pixel level: 24 months x 140 cells is enough samples to read on one window,
            # which the region-mean version (2 samples per calendar month) is not.
            acc = MonthlySpreadAccumulator("pixel")
            acc.add(members, truth, np.asarray(months).reshape(-1))
            ratios = acc.ratios()
            verdict = spread_shape_verdict(ratios, BAND)
            metrics["spread_ratio_by_month"] = ratios
            metrics["rollout_spread_shape_min"] = verdict["min"]
            metrics["rollout_spread_shape_max"] = verdict["max"]
            metrics["rollout_spread_shape_in_band"] = verdict["in_band_fraction"]

    return {
        **metrics,
        "rollout_rmse": float(np.nanmean(rmse_map)),
        # The free-running twin of val/generator_rmse: the module's teacher-forced metric is the
        # mean over samples of the SPATIAL rmse, so this is the one number that can be read
        # against that curve directly. The per-pixel reduction above is the paper's.
        "rollout_rmse_spatial": float(rmse_by_step.mean()),
        "rollout_rmse_p80": float(np.nanpercentile(rmse_map, 80)),
        "rollout_rmse_max": float(np.nanmax(rmse_map)),
        "rollout_smape": float(np.nanmean(smape_map)),
        "rollout_smape_max": float(np.nanmax(smape_map)),
        "rollout_rmse_first_step": float(rmse_by_step[0]),
        "rollout_rmse_last_step": float(rmse_by_step[-1]),
        # How much the error grows from the first free-running month to the last. A model that
        # has learned dynamics rather than persistence keeps this close to 1.
        "rollout_drift": float(rmse_by_step[-1] / (rmse_by_step[0] + 1e-8)),
        "rmse_by_step": rmse_by_step,
    }


def score_rollout(
    model: LightningModule,
    dataset: SWIDataset,
    statistics: dict[str, np.ndarray],
    num_members: int = 8,
    noise_std: float = 1.0,
    num_steps: int | None = None,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> dict[str, float | np.ndarray]:
    """Roll ``model`` forward over ``dataset`` and summarise the result. See the two above."""
    trajectories, y_true, mask = free_running_rollout(
        model=model,
        dataset=dataset,
        statistics=statistics,
        num_members=num_members,
        noise_std=noise_std,
        num_steps=num_steps,
        seed=seed,
        device=device,
    )
    horizon = trajectories.shape[1]
    months = np.array([int(dataset[i]["timestamps"].reshape(-1)[0]) for i in range(horizon)])
    return summarise_rollout(
        trajectories,
        y_true,
        mask,
        months=months,
        target_std=_single_channel_statistic(statistics, "targets_std"),
    )
