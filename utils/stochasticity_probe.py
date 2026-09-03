"""Measure how much of the ensemble's spread is actually driven by the noise vector.

The question this answers is "is this generator stochastic, and if so, from where" -- which a
``noise_std`` sweep on its own cannot answer, because the generator has *two* independent
sources of randomness at inference time and ``noise_std`` scales only one of them:

  1. the bottleneck vector ``z``, drawn in ``inference.py:180`` and scaled by ``noise_std``,
     concatenated onto the encoder's last feature map (``unet_generator.py:54``);
  2. per-block noise injection, drawn by ``torch.randn`` *inside* the forward pass at
     ``unet_frame_encoder.py:85`` and ``unet_frame_decoder.py:109`` and scaled by the learned
     ``noise_weights1/2`` parameters. This one is not affected by ``noise_std`` and is not
     switched off by ``model.eval()`` -- ``torch.randn`` fires in either mode. (The gate is
     present but commented out at ``unet_frame_decoder.py:106``.)

So the arms below cross ``noise_std`` with an explicit control on source 2:

  independent  injection noise drawn per ensemble member -- what ships today
  shared       one draw per rollout step, reused by every member -- removes source 2 from the
               across-member variance while leaving the model in its normal operating regime
  off          ``noise_weights`` zeroed -- removes source 2 entirely

``(noise_std=0, injection=off)`` is the determinism control: every member must come back
bit-identical, which is what makes the other arms' spreads readable.

Everything here is inference-only. Run with the same overrides as ``swigan/inference.py``.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from swigan.engines.swigan_lit import TTTSWIGAN
from swigan.inference import compute_trajectory_iterative
from utils.metrics import compute_rmse_per_pixel, compute_smape_per_pixel
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)

logger = logging.getLogger(__file__)

# (label, noise_std, injection mode). Ordered so the determinism control runs first: if it
# fails there is no point reading anything below it.
ARMS: list[tuple[str, float, str]] = [
    ("deterministic control  z=0, inj off", 0.0, "off"),
    ("z only          std=1  inj shared", 1.0, "shared"),
    ("z only          std=5  inj shared", 5.0, "shared"),
    ("z only          std=20 inj shared", 20.0, "shared"),
    ("injection only  z=0    inj indep", 0.0, "independent"),
    ("ships today     std=1  inj indep", 1.0, "independent"),
    ("both            std=5  inj indep", 5.0, "independent"),
    ("both            std=20 inj indep", 20.0, "independent"),
]

# Repeat draws of the shipping arm, to put an error bar on every number in the table.
REPEAT_SEEDS = [0, 1, 2]


@contextmanager
def shared_injection(num_members: int):
    """Force the in-forward noise injection to be identical across ensemble members.

    The two injection sites draw ``torch.randn((batch, 1, H, W))``. Drawing ``(1, 1, H, W)``
    and broadcasting keeps each member's forward pass statistically normal while removing
    injection from the *across-member* variance, which is the quantity being decomposed.

    In eval mode these two sites are the only ``torch.randn`` calls on the generator's forward
    path -- ``spatial_dropout`` is 0.0, ``stochastic_depth`` short-circuits when
    ``training=False``, and ``z`` is supplied rather than drawn -- so patching the global is
    scoped tightly enough to be safe. The shape guard makes that explicit.
    """
    original = torch.randn

    def patched(*size, **kwargs):
        one_arg = len(size) == 1 and isinstance(size[0], tuple | list | torch.Size)
        shape = tuple(size[0]) if one_arg else size
        if len(shape) == 4 and shape[0] == num_members and shape[1] == 1:
            return original((1, *shape[1:]), **kwargs).expand(shape).contiguous()
        return original(*size, **kwargs)

    torch.randn = patched
    try:
        yield
    finally:
        torch.randn = original


@contextmanager
def injection_off(model: torch.nn.Module):
    """Zero every ``noise_weights`` parameter, so ``out + noise * 0 == out`` exactly."""
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
                param.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "noise_weights" in name:
                    param.copy_(saved[name])


def run_arm(
    model,
    noise_std: float,
    injection: str,
    seed: int,
    num_members: int,
    feature_maps: np.ndarray,
    starting_target_maps: np.ndarray,
    timestamps: np.ndarray,
    mask: np.ndarray,
    device: str,
    dtype: str,
) -> np.ndarray:
    """Generate one ensemble under one arm. Returns (members, dates, H, W)."""
    num_steps = len(feature_maps)

    # z from a dedicated generator so the arms' z draws line up regardless of how many
    # numbers the injection sites consume from the global stream.
    z_gen = torch.Generator().manual_seed(1000 + seed)
    z = torch.randn(num_steps, num_members, model.hparams.z_dim, generator=z_gen) * noise_std

    torch.manual_seed(seed)  # drives the in-forward injection draws
    kwargs = {
        "model": model,
        "inputs_maps": torch.tensor(feature_maps * mask.squeeze(), device=device).float(),
        "starting_target_maps": torch.tensor(
            starting_target_maps * mask.squeeze(), device=device
        ).float(),
        "input_timestamps": torch.tensor(timestamps.squeeze(), device=device).long(),
        "mask": torch.tensor(mask, device=device).float(),
        "z": z.to(device),
        "dtype": np.dtype(dtype),
    }

    if injection == "off":
        with injection_off(model):
            traj = compute_trajectory_iterative(**kwargs)
    elif injection == "shared":
        with shared_injection(num_members):
            traj = compute_trajectory_iterative(**kwargs)
    elif injection == "independent":
        traj = compute_trajectory_iterative(**kwargs)
    else:
        raise ValueError(f"unknown injection mode {injection!r}")

    return traj.squeeze(2).astype(np.float32)  # drop the singleton channel


def summarise(
    traj: np.ndarray, y_true: np.ndarray, mask: np.ndarray, target_std: float
) -> dict[str, float | np.ndarray]:
    """Diversity and accuracy statistics for one ensemble.

    ``traj`` is (members, dates, H, W); ``mask`` is a boolean (H, W).
    """
    members = traj[:, :, mask]  # (M, T, P)

    # Across-member spread, per date and pixel, then pooled. Reported in standardised units
    # so it is comparable with the temporal variation of the signal itself.
    spread_map = members.std(axis=0, ddof=1)  # (T, P)
    spread = float(spread_map.mean()) / target_std
    spread_by_step = spread_map.mean(axis=1) / target_std  # (T,)

    # The signal the spread has to be judged against: how much the ensemble mean itself moves
    # across the 24 months, per pixel, averaged over pixels.
    ens_mean = members.mean(axis=0)  # (T, P)
    signal = float(ens_mean.std(axis=0, ddof=1).mean()) / target_std

    # Determinism check: largest disagreement between any member and the ensemble mean.
    max_dev = float(np.abs(members - ens_mean[None]).max())

    # How much of the observation the ensemble actually brackets. A point forecast scores ~0.
    truth = y_true[:, mask]  # (T, P)
    inside = (truth >= members.min(axis=0)) & (truth <= members.max(axis=0))
    coverage = float(inside.mean())

    # Effective number of distinct trajectories: participation ratio of the PCA spectrum of
    # the member deviations. 1.0 means every member differs along the same single direction.
    dev = (members - ens_mean[None]).reshape(members.shape[0], -1)
    if max_dev == 0.0:
        eff_rank = 0.0
    else:
        sv = np.linalg.svd(dev, compute_uv=False)
        var = sv**2
        eff_rank = float(var.sum() ** 2 / (var**2).sum())

    # The paper's metric, per arm, so diversity can be read against accuracy.
    rmse = np.nanmean(
        np.stack([compute_rmse_per_pixel(y_true, traj[m], mask) for m in range(traj.shape[0])]),
        axis=0,
    )
    smape = np.nanmean(
        np.stack([compute_smape_per_pixel(y_true, traj[m], mask) for m in range(traj.shape[0])]),
        axis=0,
    )
    return {
        "spread": spread,
        "spread_by_step": spread_by_step,
        "signal": signal,
        "spread_ratio": spread / signal,
        "max_dev": max_dev,
        "coverage": coverage,
        "eff_rank": eff_rank,
        "rmse_max": float(np.nanmax(rmse)),
        "rmse_p80": float(np.nanpercentile(rmse, 80)),
        "rmse_mean": float(np.nanmean(rmse)),
        "smape_max": float(np.nanmax(smape)),
        "under_paper_p80": float(np.nanmean(rmse[mask] < 0.16)),
    }


def prepare(cfg: DictConfig) -> dict:
    """Load the checkpoint and build the rollout inputs.

    Mirrors ``swigan/inference.py:80-174`` for the inputs and ``evaluate.py:98-104`` for the
    observed maps, so the probe scores the same window the paper metric does. Kept as one
    function so the other inference-only probes can reuse it.
    """
    dataset_cfg, model_cfg = cfg["dataset"], cfg["model"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    filepath = dataset_cfg["filepath"]
    if Path(filepath).suffix == ".parquet":
        input_df = pd.read_parquet(filepath)
    else:
        input_df = pd.read_csv(filepath)

    x_dim_col = dataset_cfg["x_dim_column"] or "x"
    y_dim_col = dataset_cfg["y_dim_column"] or "y"
    if dataset_cfg["fill_missing_pixels"]:
        input_df = fill_all_missing_pixels(input_df, x_dim_col=x_dim_col, y_dim_col=y_dim_col)
    input_df["mask"] = 1.0 * (input_df.loc[:, "scenario"] != 0)

    all_indices = input_df.loc[
        (input_df["year"] == float(dataset_cfg["starting_year"]))
        & (input_df["month"] == float(dataset_cfg["starting_month"]))
    ].index.tolist()
    idx = sorted(all_indices)[0]
    starting_dataset, input_dataset = input_df.iloc[:idx], input_df.iloc[idx:]

    feature_columns = dataset_cfg["input_columns"]
    target_column = dataset_cfg["target_column"]
    useful = [*feature_columns, "year", "month", x_dim_col, y_dim_col, target_column, "mask"]
    drop = [c for c in input_dataset.columns if c not in useful]
    input_dataset = coerce_comma_decimal_columns(
        input_dataset.drop(columns=drop), [*feature_columns, target_column]
    )
    starting_dataset = coerce_comma_decimal_columns(
        starting_dataset.drop(columns=drop), [*feature_columns, target_column]
    )

    model = TTTSWIGAN.load_from_checkpoint(
        model_cfg["checkpoint_path"], loss_fn="l1", device=device
    )
    model.statistics = np.load(model_cfg["train_dataset_statistics_path"])
    model.eval()  # same fix as inference.py:154

    map_height, map_width = model.hparams.input_map_dims
    feature_maps, target_maps, timestamps, mask = dataframe_to_rasters(
        input_dataset, target_column, feature_columns, map_height, map_width
    )
    _, starting_targets, _, _ = dataframe_to_rasters(
        starting_dataset, target_column, feature_columns, map_height, map_width
    )
    num_input_steps = model.hparams.input_channels - len(feature_columns)
    starting_target_maps = starting_targets[-num_input_steps:].squeeze(1)

    feats_mean, feats_std = model.statistics["feats_mean"], model.statistics["feats_std"]
    feature_maps = (feature_maps - feats_mean) / feats_std
    feature_maps = np.where(mask.squeeze(), feature_maps, 0.0)

    y_true = target_maps.squeeze(1).astype(np.float32)  # (T, H, W) observed
    bool_mask = mask.squeeze(0).astype(bool) if mask.ndim == 3 else mask.squeeze().astype(bool)
    target_std = float(model.statistics["targets_std"])
    num_members = int(model_cfg["num_trajectories"])

    logger.info(
        f"{num_members} members x {len(feature_maps)} steps, {int(bool_mask.sum())} unmasked "
        f"pixels, targets_std={target_std:.4f}"
    )

    return {
        "rollout": dict(
            model=model,
            num_members=num_members,
            feature_maps=feature_maps,
            starting_target_maps=starting_target_maps,
            timestamps=timestamps,
            mask=mask,
            device=device,
            dtype=model_cfg["dtype"],
        ),
        "model": model,
        "y_true": y_true,
        "bool_mask": bool_mask,
        "target_std": target_std,
        "num_members": num_members,
        "num_steps": len(feature_maps),
        "device": device,
    }


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run every arm and print the decomposition."""
    model_cfg = cfg["model"]
    prepared = prepare(cfg)
    shared = prepared["rollout"]
    y_true, bool_mask = prepared["y_true"], prepared["bool_mask"]
    target_std, num_members = prepared["target_std"], prepared["num_members"]
    num_steps = prepared["num_steps"]

    rows, spread_curves = [], {}
    for label, noise_std, injection in ARMS:
        traj = run_arm(noise_std=noise_std, injection=injection, seed=0, **shared)
        stats = summarise(traj, y_true, bool_mask, target_std)
        rows.append((label, stats))
        spread_curves[label] = stats["spread_by_step"]
        logger.info(f"{label:38s} spread={stats['spread']:.5f} max_dev={stats['max_dev']:.2e}")

    # Repeat the shipping arm to size the probe's own noise.
    repeats = [
        summarise(run_arm(noise_std=1.0, injection="independent", seed=s, **shared),
                  y_true, bool_mask, target_std)
        for s in REPEAT_SEEDS
    ]

    print("\n" + "=" * 108)
    print("ENSEMBLE DIVERSITY BY NOISE SOURCE".center(108))
    print(f"{Path(model_cfg['checkpoint_path']).parts[0]} · {num_members} members · "
          f"{num_steps} rollout steps · eval mode".center(108))
    print("=" * 108)
    header = (f"{'arm':38s} {'spread':>8s} {'/signal':>8s} {'eff':>5s} {'cover':>7s} "
              f"{'maxdev':>9s} {'rmse max':>9s} {'rmse mean':>9s}")
    print(header)
    print("-" * 108)
    for label, s in rows:
        print(f"{label:38s} {s['spread']:8.5f} {s['spread_ratio']:7.1%} {s['eff_rank']:5.1f} "
              f"{s['coverage']:7.1%} {s['max_dev']:9.2e} {s['rmse_max']:9.4f} "
              f"{s['rmse_mean']:9.4f}")
    print("-" * 108)
    print(f"{'ships today, 3 repeat seeds':38s} "
          f"{np.mean([r['spread'] for r in repeats]):8.5f} "
          f"sd {np.std([r['spread'] for r in repeats], ddof=1):.5f}"
          f"{'':21s}{np.mean([r['rmse_max'] for r in repeats]):9.4f} "
          f"{np.mean([r['rmse_mean'] for r in repeats]):9.4f}")
    print(f"\nspread   = across-member sd, standardised units (divided by targets_std "
          f"{target_std:.4f})")
    print(f"/signal  = that spread as a share of the ensemble mean's own variation across the "
          f"{num_steps} months")
    print("eff      = participation ratio of the member-deviation spectrum; 1.0 = every member "
          "differs along one direction")
    print("cover    = share of pixel-months where the observation falls inside the ensemble's "
          "min-max envelope")
    print("maxdev   = largest deviation of any member from the ensemble mean, raw SWI units "
          "(0 = deterministic)")
    print("=" * 108 + "\n")

    out = Path(model_cfg["saving_path"]).with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        f"{out}_stochasticity.npz",
        arms=np.array([label for label, _ in rows]),
        **{
            key: np.array([s[key] for _, s in rows])
            for key in ("spread", "signal", "spread_ratio", "max_dev", "coverage", "eff_rank",
                        "rmse_max", "rmse_p80", "rmse_mean", "smape_max", "under_paper_p80")
        },
        spread_by_step=np.stack([spread_curves[label] for label, _ in rows]),
        repeat_spread=np.array([r["spread"] for r in repeats]),
        repeat_rmse_max=np.array([r["rmse_max"] for r in repeats]),
        target_std=target_std,
    )
    logger.info(f"Saved to '{out}_stochasticity.npz'.")


if __name__ == "__main__":
    main()
