"""Plot the maps of a saved inference rollout next to the observed ones, month by month.

The layout is the one of the training figures (``utils.callbacks.RandomTimeStepMapCallback``):
the observed map on top, the generated one below, one shared colour scale per month, RdYlGn.
An inference run is an ensemble rather than one prediction, so each month gets four rows:
the observation, the ensemble mean, one member, and the spread across members (its own scale).

    python -m utils.plot_inference_maps
    python -m utils.plot_inference_maps +plot.years=[2024,2025] +plot.member=5

Reads the same ``config/inference_swigan.yaml`` the inference ran with, so the observed maps
come from the same file, the same starting date and the same pixel filling. Members
``0 .. neutral_members-1`` of a dispersion spec see a neutral history; the default member is
the first one that does not.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from omegaconf import DictConfig

from utils.callbacks import MONTHS
from utils.dispersion import load_dispersion_spec
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
)

logger = logging.getLogger(__file__)


def observed_maps(
    dataset_cfg: DictConfig, height: int, width: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Observed SWI maps from the starting date on, the region mask and the (year, month) list.

    Mirrors the preprocessing of ``swigan.inference.main`` up to ``dataframe_to_rasters``.
    """
    x_dim_col = dataset_cfg["x_dim_column"] or "x"
    y_dim_col = dataset_cfg["y_dim_column"] or "y"
    df = pd.read_parquet(dataset_cfg["filepath"])
    if dataset_cfg["fill_missing_pixels"]:
        df = fill_all_missing_pixels(df, x_dim_col=x_dim_col, y_dim_col=y_dim_col)
    df["mask"] = 1.0 * (df.loc[:, "scenario"] != 0)

    start = df.loc[
        (df["year"] == float(dataset_cfg["starting_year"]))
        & (df["month"] == float(dataset_cfg["starting_month"]))
    ].index.min()
    df = df.iloc[start:]
    target = dataset_cfg["target_column"]
    features = list(dataset_cfg["input_columns"])
    df = coerce_comma_decimal_columns(df, [*features, target])

    _, targets, _, mask = dataframe_to_rasters(
        df, target, features, height, width, x_dim_col, y_dim_col
    )
    # The same complete-month filter dataframe_to_rasters applies, to date each map.
    counts = df.groupby(["year", "month"]).size()
    calendar = [(int(y), int(m)) for (y, m), n in counts.items() if n == height * width]
    return targets[:, 0], mask.squeeze().astype(bool), calendar


def plot_months(
    truth: np.ndarray,
    members: np.ndarray,
    mask: np.ndarray,
    calendar: list[tuple[int, int]],
    member: int,
    title: str,
    path: Path,
) -> None:
    """One figure, one column per month: observed, ensemble mean, one member, spread."""
    num_months = len(calendar)
    fig, axes = plt.subplots(4, num_months, figsize=(5 * num_months, 12), squeeze=False)
    plt.subplots_adjust(wspace=0.5)
    ens_mean = members.mean(axis=0)
    ens_std = members.std(axis=0)
    for k, (year, month) in enumerate(calendar):
        label = f"{MONTHS[month - 1]} {year}"
        rows = [
            (f"GT {label}", truth[k]),
            (f"Ensemble mean {label}", ens_mean[k]),
            (f"Member {member} {label}", members[member, k]),
        ]
        merged = np.stack([field[mask] for _, field in rows])
        norm = colors.Normalize(vmin=merged.min(), vmax=merged.max())
        for row, (name, field) in enumerate(rows):
            axes[row, k].set_title(name)
            p = axes[row, k].imshow(np.where(mask, field, np.nan), cmap="RdYlGn", norm=norm)
        plt.colorbar(p, ax=axes[:3, k], shrink=0.4)
        axes[3, k].set_title(f"Ensemble sd {label}")
        p = axes[3, k].imshow(np.where(mask, ens_std[k], np.nan), cmap="viridis")
        plt.colorbar(p, ax=axes[3, k], shrink=0.8)
    fig.suptitle(title)
    plt.savefig(path, dpi=200)
    plt.close(fig)


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Plot every month of the requested years, six months per figure."""
    dataset_cfg, model_cfg = cfg["dataset"], cfg["model"]
    plot_cfg = cfg.get("plot") or {}

    saving_path = Path(model_cfg["saving_path"])
    npy_path = saving_path.with_suffix(".npy")
    members = np.load(npy_path).astype(np.float32)[:, :, 0]  # (M, T, H, W), SWI
    height, width = members.shape[-2:]
    truth, mask, calendar = observed_maps(dataset_cfg, height, width)
    if len(calendar) != members.shape[1]:
        raise ValueError(
            f"{npy_path} holds {members.shape[1]} months but the dataset gives {len(calendar)} "
            "from the starting date: was it produced with this config?"
        )

    dispersion = model_cfg.get("dispersion")
    neutral = load_dispersion_spec(dispersion)["neutral_members"] if dispersion else 0
    member = int(plot_cfg.get("member", min(neutral, len(members) - 1)))
    years = [int(y) for y in plot_cfg.get("years", [2024, 2025])]
    months_per_plot = int(plot_cfg.get("months_per_plot", 6))

    out_dir = Path(plot_cfg.get("directory", f"{saving_path}_figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [i for i, (y, _) in enumerate(calendar) if y in years]
    for start in range(0, len(selected), months_per_plot):
        idx = selected[start : start + months_per_plot]
        first, last = calendar[idx[0]], calendar[idx[-1]]
        name = f"inference_{first[0]}-{first[1]:02d}_{last[0]}-{last[1]:02d}.png"
        plot_months(
            truth[idx],
            members[:, idx],
            mask,
            [calendar[i] for i in idx],
            member,
            f"Inference maps {npy_path.name}, {len(members)} members",
            out_dir / name,
        )
        logger.info(f"Saved {out_dir / name}")


if __name__ == "__main__":
    main()
