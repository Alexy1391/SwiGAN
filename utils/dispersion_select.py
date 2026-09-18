"""Choose the state-dispersion instrument on windows that end before the forecast, then ship it.

WHY THIS EXISTS. ``utils/noise_gain_transfer.py`` settled the protocol: a parameter chosen by
the score it is quoted against is a ceiling, not a result, so the parameter has to be picked on
calibration windows that end before the scored window opens, and the transfer measured. The
same rule applies to the two state knobs of :mod:`utils.dispersion` and to the per-month offset,
and this module applies it. Nothing below is selected on the test window; the test window is
scored once, at the end, with parameters fixed beforehand.

WHAT IS SELECTED. A grid over ``neutral_members`` x ``state_sd``, one 20-member rollout per
calibration window per point. For every grid point the seasonal offset is fitted LEAVE-ONE-
WINDOW-OUT inside the calibration set -- fitted on thirteen windows, applied to the fourteenth --
so the CRPS the selection reads is the CRPS of the corrected ensemble on a window the correction
never saw. The point with the lowest pooled corrected CRPS wins; the raw argmin is reported
beside it. The offset that ships is then refitted on all calibration windows for the winner.

WHAT IS REPORTED ON TEST. Four arms, three rollout seeds each: the shipped ensemble, the
shipped ensemble plus the offset, the selected point, the selected point plus the offset --
CRPS, the ensemble mean's RMSE, spread-skill, envelope coverage, bias and reliability. And the
acceptance test of :mod:`utils.spread_shape`: the twelve monthly spread / required ratios,
pooled over the calibration windows at region level and at pixel level, with the pass/fail
against ``band``. A fixed perturbation is a fixed width, so expect the annual spread-skill to
pass and the monthly test to fail on the summer months; that failure is the training
objective's to fix, not this tool's.

THE WINDOWS. One continuous frame is built from the listed parquets (train 1960-2021 and the
2022-2025 file for Corse), so every January-start window from 2008 to 2021 can be rolled out
from its own eight months of observed history exactly as the shipping inference seeds itself.
Windows starting before the end of the training split are in-sample for the weights; they are
kept, as the gain-transfer tool keeps them, because three post-training windows is the sample
size that misled the earlier rounds. The 2022 window (val) is scored as an out-of-sample check
that is also never fitted on.

The output JSON is what ``model.dispersion`` in ``config/inference_swigan.yaml`` points at.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from swigan.engines.swigan_lit import TTTSWIGAN
from swigan.inference import compute_trajectory_iterative
from utils.dispersion import (
    apply_seasonal_offsets,
    dispersion_spec,
    fit_seasonal_offsets,
    member_perturbations,
    month_index,
    save_dispersion_spec,
)
from utils.ensemble_scores import score
from utils.inference_modes import dropout_active
from utils.preprocessing import (
    coerce_comma_decimal_columns,
    dataframe_to_rasters,
    fill_all_missing_pixels,
    single_channel_statistic,
    standardize_swi_history,
)
from utils.spread_shape import (
    MONTH_LABELS,
    MonthlySpreadAccumulator,
    format_monthly_table,
    spread_shape_verdict,
)

logger = logging.getLogger(__file__)

STEPS = 24
SHIPPED: tuple[int, float] = (0, 0.0)


def load_continuous_frame(
    filepaths: list[str], x_dim_col: str, y_dim_col: str, fill_missing_pixels: bool
) -> pd.DataFrame:
    """Concatenate the parquets into one chronological frame, one row per cell and month.

    Where two files hold the same month the first file listed wins, so ``[train, 2022-2025]``
    takes 2021 from the training file. The mask column keys off ``scenario`` as every other
    loader does, after the missing cells have been filled.
    """
    frames = [pd.read_parquet(path) for path in filepaths]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["year", "month", x_dim_col, y_dim_col], keep="first")
    if fill_missing_pixels:
        frame = fill_all_missing_pixels(frame, x_dim_col=x_dim_col, y_dim_col=y_dim_col)
    frame["mask"] = 1.0 * (frame.loc[:, "scenario"] != 0)
    return frame.sort_values(["year", "month", y_dim_col, x_dim_col]).reset_index(drop=True)


def raster_calendar(frame: pd.DataFrame, height: int, width: int) -> list[tuple[int, int]]:
    """List the ``(year, month)`` of every raster ``dataframe_to_rasters`` keeps, in its order."""
    counts = frame.groupby(["year", "month"]).size()
    calendar = []
    for year in sorted(frame["year"].unique()):
        for month in sorted(frame["month"].unique()):
            if counts.get((year, month), 0) == height * width:
                calendar.append((int(year), int(month)))
    return calendar


class WindowedRasters:
    """Every rollout input for any 24-month window of one continuous frame, built once."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        height: int,
        width: int,
        statistics: dict[str, np.ndarray],
        num_input_steps: int,
        x_dim_col: str = "x",
        y_dim_col: str = "y",
    ) -> None:
        """Rasterize and standardize the frame the way the shipping inference path does."""
        useful = [*feature_columns, "year", "month", x_dim_col, y_dim_col, target_column, "mask"]
        frame = frame.drop(columns=[c for c in frame.columns if c not in useful])
        frame = coerce_comma_decimal_columns(frame, [*feature_columns, target_column])
        feats, targets, timestamps, mask = dataframe_to_rasters(
            frame, target_column, feature_columns, height, width, x_dim_col, y_dim_col
        )
        self.calendar = raster_calendar(frame, height, width)
        if len(self.calendar) != len(feats):
            raise RuntimeError("calendar and rasters disagree on the months kept")
        self.mask = mask.astype(np.float32)  # (1, H, W)
        self.bool_mask = mask.squeeze(0).astype(bool)
        feats = (feats - statistics["feats_mean"]) / statistics["feats_std"]
        self.feats = np.where(self.bool_mask, feats, 0.0).astype(np.float32)
        self.truth = (targets.squeeze(1) * self.bool_mask).astype(np.float32)  # (T, H, W)
        self.history = standardize_swi_history(targets.squeeze(1), statistics, mask).astype(
            np.float32
        )
        self.timestamps = timestamps
        self.num_input_steps = num_input_steps

    def window(self, year: int, month: int = 1, steps: int = STEPS) -> dict[str, Any]:
        """Build the inputs and observations of the window opening at ``(year, month)``."""
        try:
            start = self.calendar.index((year, month))
        except ValueError as error:
            raise ValueError(f"{year}-{month:02d} is not a complete month of the frame") from error
        if start < self.num_input_steps:
            raise ValueError(
                f"{year}-{month:02d} has fewer than {self.num_input_steps} months before it"
            )
        if start + steps > len(self.calendar):
            raise ValueError(f"{year}-{month:02d} + {steps} months runs past the frame")
        stop = start + steps
        last_year, last_month = self.calendar[stop - 1]
        return {
            "label": f"{year}-{month:02d}..{last_year}-{last_month:02d}",
            "feats": self.feats[start:stop],
            "history": self.history[start - self.num_input_steps : start],
            "timestamps": self.timestamps[start:stop],
            "months": month_index(self.timestamps[start:stop]),
            "truth": self.truth[start:stop][:, self.bool_mask],  # (T, P)
        }


def rollout_window(
    model: TTTSWIGAN,
    rasters: WindowedRasters,
    window: dict[str, Any],
    num_members: int,
    neutral_members: int,
    state_sd: float,
    member_seed: int,
    seed: int,
    noise_std: float,
    device: torch.device | str,
) -> np.ndarray:
    """One ensemble over one window through the shipping rollout. Returns ``(M, T, P)``, SWI."""
    scale, shift = member_perturbations(num_members, neutral_members, state_sd, member_seed)
    z = (
        torch.randn(
            len(window["feats"]),
            num_members,
            model.hparams.z_dim,
            generator=torch.Generator().manual_seed(1000 + seed),
        )
        * noise_std
    )
    torch.manual_seed(seed)  # the in-forward injection draws
    trajectories = compute_trajectory_iterative(
        model=model,
        inputs_maps=torch.tensor(window["feats"], device=device),
        starting_target_maps=torch.tensor(window["history"], device=device),
        input_timestamps=torch.tensor(window["timestamps"].squeeze(), device=device).long(),
        mask=torch.tensor(rasters.mask, device=device),
        z=z.to(device),
        dtype=np.float32,
        history_scale=scale,
        history_shift=shift,
    )
    return trajectories[:, :, 0][:, :, rasters.bool_mask]


def window_metrics(members: np.ndarray, truth: np.ndarray, target_std: float) -> dict[str, float]:
    """Score one arm over one window on the six numbers it is judged on."""
    scored = score(members, truth, target_std)
    ens_mean = members.mean(axis=0)
    return {
        "crps": scored["crps"],
        "rmse": float(np.sqrt(((ens_mean - truth) ** 2).mean())),
        "spread_skill": scored["spread_skill"],
        "coverage": scored["coverage"],
        "bias": float((ens_mean - truth).mean()),
        # Pooled over windows the signed bias of a leave-one-out corrected ensemble is 0 by
        # construction; the magnitude per window is what says whether the offset transfers.
        "abs_bias": float(abs((ens_mean - truth).mean())),
        "reliability": scored["reliability"],
    }


def pooled(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric over a list of per-window (or per-seed) rows."""
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def acceptance(
    windows: list[dict[str, Any]],
    ensembles: list[list[np.ndarray]],
    band: tuple[float, float],
    offsets: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pool region- and pixel-level monthly ratios over ``windows`` and return the verdicts.

    ``ensembles[w]`` holds one ``(M, T, P)`` array per rollout seed of window ``w``; every seed
    adds its samples to the pool.
    """
    out: dict[str, Any] = {}
    for level in ("region", "pixel"):
        acc = MonthlySpreadAccumulator(level)
        for window, per_seed in zip(windows, ensembles, strict=True):
            for members in per_seed:
                if offsets is not None:
                    members = apply_seasonal_offsets(members, window["months"], offsets)
                acc.add(members, window["truth"], window["months"])
        ratios = acc.ratios()
        out[level] = {
            "ratios": ratios.tolist(),
            "produced": acc.produced().tolist(),
            "required": acc.required().tolist(),
            **spread_shape_verdict(ratios, band),
        }
    return out


Point = tuple[int, float]
Members = dict[Point, list[list[np.ndarray]]]  # point -> window -> seed -> (M, T, P)


def evaluate_grid(
    grid: list[Point],
    calib: list[dict[str, Any]],
    seeds: list[int],
    run: Any,
    target_std: float,
) -> tuple[Members, dict[Point, dict[str, dict[str, float]]]]:
    """Roll every grid point over every calibration window; score raw and leave-one-out corrected.

    The seasonal offset is fitted on the other windows' ensemble means (averaged over seeds,
    which is still an ensemble mean) and applied to each seed's members of the held-out
    window, so the corrected CRPS is that of a correction the window never saw. Metrics are
    averaged over seeds, never the members themselves.
    """
    members: Members = {}
    raw: dict[Point, list[dict[str, float]]] = {}
    for k, sd in grid:
        members[(k, sd)] = []
        raw[(k, sd)] = []
        for window in calib:
            per_seed = [run(window, k, sd, seed) for seed in seeds]
            members[(k, sd)].append(per_seed)
            raw[(k, sd)].append(
                pooled([window_metrics(m, window["truth"], target_std) for m in per_seed])
            )
        logger.info(f"  k={k} sd={sd}: pooled raw CRPS {pooled(raw[(k, sd)])['crps']:.4f}")

    corrected: dict[Point, list[dict[str, float]]] = {}
    for point, per_window in members.items():
        ens_means = [np.mean([m.mean(axis=0) for m in per_seed], axis=0) for per_seed in per_window]
        corrected[point] = []
        for hold, window in enumerate(calib):
            others = [i for i in range(len(calib)) if i != hold]
            offsets = fit_seasonal_offsets(
                [ens_means[i] for i in others],
                [calib[i]["truth"] for i in others],
                [calib[i]["months"] for i in others],
            )
            rows = [
                window_metrics(
                    apply_seasonal_offsets(m, window["months"], offsets),
                    window["truth"],
                    target_std,
                )
                for m in per_window[hold]
            ]
            corrected[point].append(pooled(rows))
    table = {p: {"raw": pooled(raw[p]), "corrected": pooled(corrected[p])} for p in grid}
    return members, table


def offsets_for(members: Members, point: Point, calib: list[dict[str, Any]]) -> np.ndarray:
    """Refit the seasonal offset of one grid point on every calibration window."""
    ens_means = [np.mean([m.mean(axis=0) for m in per_seed], axis=0) for per_seed in members[point]]
    return fit_seasonal_offsets(
        ens_means, [w["truth"] for w in calib], [w["months"] for w in calib]
    )


def report_grid(
    grid: list[Point],
    table: dict[Point, dict[str, dict[str, float]]],
    best: Point,
    best_raw: Point,
    title: str,
) -> None:
    """Print the grid table, pooled over the calibration windows."""
    print("\n" + "=" * 118)
    print(title.center(118))
    print("=" * 118)
    print(
        f"{'k':>3s} {'sd':>5s} | {'raw CRPS':>9s} {'vs ship':>8s} {'RMSE':>7s} {'s/s':>6s} "
        f"{'cover':>6s} | {'corr CRPS':>9s} {'vs ship':>8s} {'RMSE':>7s} {'s/s':>6s} "
        f"{'cover':>6s} {'|bias|':>7s}"
    )
    ship_raw = table[SHIPPED]["raw"]["crps"]
    ship_corr = table[SHIPPED]["corrected"]["crps"]
    for k, sd in grid:
        r, c = table[(k, sd)]["raw"], table[(k, sd)]["corrected"]
        flag = ""
        if (k, sd) == best:
            flag = " <- selected"
        elif (k, sd) == best_raw:
            flag = " <- raw argmin"
        print(
            f"{k:3d} {sd:5.2f} | {r['crps']:9.4f} {100 * (r['crps'] / ship_raw - 1):+7.1f}% "
            f"{r['rmse']:7.4f} {r['spread_skill']:6.3f} {100 * r['coverage']:5.1f}% | "
            f"{c['crps']:9.4f} {100 * (c['crps'] / ship_corr - 1):+7.1f}% {c['rmse']:7.4f} "
            f"{c['spread_skill']:6.3f} {100 * c['coverage']:5.1f}% {c['abs_bias']:7.4f}{flag}"
        )


def score_transfer(
    name: str,
    window: dict[str, Any],
    arms: list[tuple[str, Point, np.ndarray | None]],
    run: Any,
    seeds: list[int],
    target_std: float,
) -> dict[str, Any]:
    """Score the fixed arms on one window the selection never saw, and print the table."""
    print("\n" + "=" * 118)
    print(
        f"{name.upper()}  {window['label']}  -- {len(seeds)} rollout seeds, every parameter "
        "fixed on the calibration windows".center(118)
    )
    print("=" * 118)
    print(
        f"{'arm':<28s} {'CRPS':>8s} {'vs ship':>8s} {'RMSE mean':>10s} {'vs ship':>8s} "
        f"{'s/s':>7s} {'cover':>7s} {'bias':>8s} {'reliab':>7s}"
    )
    result: dict[str, Any] = {"window": window["label"], "arms": {}}
    base: dict[str, float] | None = None
    for label, point, offsets in arms:
        rows = []
        for seed in seeds:
            ens = run(window, point[0], point[1], seed)
            if offsets is not None:
                ens = apply_seasonal_offsets(ens, window["months"], offsets)
            rows.append(window_metrics(ens, window["truth"], target_std))
        m = pooled(rows)
        crps_values = [r["crps"] for r in rows]
        m["crps_sd_over_seeds"] = float(np.std(crps_values, ddof=1)) if len(rows) > 1 else 0.0
        result["arms"][label] = m
        base = base or m
        print(
            f"{label:<28s} {m['crps']:8.4f} {100 * (m['crps'] / base['crps'] - 1):+7.1f}% "
            f"{m['rmse']:10.4f} {100 * (m['rmse'] / base['rmse'] - 1):+7.1f}% "
            f"{m['spread_skill']:7.3f} {100 * m['coverage']:6.1f}% {m['bias']:+8.4f} "
            f"{m['reliability']:7.3f}"
        )
    return result


def report_acceptance(
    calib: list[dict[str, Any]],
    members: Members,
    arms: list[tuple[str, Point, np.ndarray | None]],
    band: tuple[float, float],
) -> dict[str, Any]:
    """Print the monthly spread / required tables and verdicts, pooled over the calibration set."""
    print("\n" + "=" * 118)
    print(
        f"ACCEPTANCE: monthly spread / required, pooled over the {len(calib)} calibration "
        f"windows; band {band[0]}-{band[1]}".center(118)
    )
    print("=" * 118)
    accept: dict[str, Any] = {}
    for label, point, offsets in arms:
        accept[label] = acceptance(calib, members[point], band, offsets)
        for level in ("region", "pixel"):
            a = accept[label][level]
            print(
                format_monthly_table(
                    {
                        "required (rms error)": a["required"],
                        "produced (spread)": a["produced"],
                        "ratio": a["ratios"],
                    },
                    header=f"{label} · {level}",
                )
            )
            verdict = "PASS" if a["pass"] else "FAIL"
            print(
                f"{'':<28s}-> {verdict}: {a['n_in_band']}/{a['n_months']} months in band, "
                f"range {a['min']:.2f}-{a['max']:.2f}, worst {a['worst_month']}\n"
            )
    return accept


@hydra.main(config_path="../config", config_name="dispersion_select", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Select on the calibration windows, score the transfer, write the spec."""
    dataset_cfg, model_cfg, sel = cfg["dataset"], cfg["model"], cfg["selection"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x_dim_col = dataset_cfg["x_dim_column"] or "x"
    y_dim_col = dataset_cfg["y_dim_column"] or "y"

    model = TTTSWIGAN.load_from_checkpoint(
        model_cfg["checkpoint_path"], loss_fn="l1", device=device
    )
    model.statistics = dict(np.load(model_cfg["train_dataset_statistics_path"]))
    model.eval()
    target_std = single_channel_statistic(model.statistics, "targets_std")
    height, width = model.hparams.input_map_dims
    num_input_steps = model.hparams.input_channels - len(dataset_cfg["input_columns"])

    frame = load_continuous_frame(
        list(dataset_cfg["filepaths"]), x_dim_col, y_dim_col, dataset_cfg["fill_missing_pixels"]
    )
    rasters = WindowedRasters(
        frame,
        list(dataset_cfg["input_columns"]),
        dataset_cfg["target_column"],
        height,
        width,
        model.statistics,
        num_input_steps,
        x_dim_col,
        y_dim_col,
    )
    logger.info(
        f"Frame {rasters.calendar[0]} .. {rasters.calendar[-1]}, {len(rasters.calendar)} "
        f"months, {int(rasters.bool_mask.sum())} cells; targets_std {target_std:.4f}"
    )

    num_members = int(model_cfg["num_trajectories"])
    noise_std = float(model_cfg["noise_std"])
    member_seed = int(sel["member_seed"])
    dropout = bool(model_cfg["dropout"])
    band = (float(sel["band"][0]), float(sel["band"][1]))
    grid: list[Point] = list(
        itertools.product(
            [int(k) for k in sel["neutral_members"]], [float(s) for s in sel["state_sd"]]
        )
    )
    if SHIPPED not in grid:
        raise ValueError(
            "the grid must contain neutral_members=0, state_sd=0 -- the shipped ensemble"
        )
    calib_years = [int(y) for y in sel["calibration_years"]]
    calib = [rasters.window(y) for y in calib_years]
    holdout = rasters.window(int(sel["holdout_year"]))
    test = rasters.window(int(sel["test_year"]))
    selection_seeds = [int(s) for s in sel["selection_seeds"]]
    evaluation_seeds = [int(s) for s in sel["evaluation_seeds"]]

    def run(window: dict[str, Any], k: int, sd: float, seed: int) -> np.ndarray:
        with dropout_active(model, dropout):
            return rollout_window(
                model, rasters, window, num_members, k, sd, member_seed, seed, noise_std, device
            )

    logger.info(
        f"{len(grid)} grid points x {len(calib)} calibration windows x "
        f"{len(selection_seeds)} seed(s)"
    )
    members, table = evaluate_grid(grid, calib, selection_seeds, run, target_std)
    best = min(grid, key=lambda p: table[p]["corrected"]["crps"])
    best_raw = min(grid, key=lambda p: table[p]["raw"]["crps"])
    report_grid(
        grid,
        table,
        best,
        best_raw,
        f"GRID ON {len(calib)} CALIBRATION WINDOWS {calib_years[0]}-{calib_years[-1]}, pooled; "
        "'corrected' = seasonal offset fitted leave-one-window-out",
    )

    offsets = {
        SHIPPED: offsets_for(members, SHIPPED, calib),
        best: offsets_for(members, best, calib),
    }
    selected_label = f"selected k={best[0]} sd={best[1]}"
    arms: list[tuple[str, Point, np.ndarray | None]] = [
        ("shipped", SHIPPED, None),
        ("shipped + offset", SHIPPED, offsets[SHIPPED]),
        (selected_label, best, None),
        ("selected + offset", best, offsets[best]),
    ]
    transfer = {
        name: score_transfer(name, window, arms, run, evaluation_seeds, target_std)
        for name, window in (("holdout (never fitted on)", holdout), ("TEST", test))
    }
    accept = report_acceptance(calib, members, [arms[0], arms[2], arms[3]], band)

    out_dir = Path(cfg["output"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = dispersion_spec(
        best[0],
        best[1],
        offsets[best],
        member_seed,
        offsets_units="SWI",
        checkpoint=str(model_cfg["checkpoint_path"]),
        selected_on=(
            f"{len(calib)} January-start windows {calib_years[0]}-{calib_years[-1]}, "
            "argmin of pooled leave-one-window-out corrected CRPS"
        ),
        inference_dropout=dropout,
        num_trajectories=num_members,
    )
    spec_path = out_dir / "state_dispersion.json"
    save_dispersion_spec(spec, spec_path)
    results = {
        "grid": [
            {
                "neutral_members": k,
                "state_sd": sd,
                **{f"raw_{a}": b for a, b in table[(k, sd)]["raw"].items()},
                **{f"corrected_{a}": b for a, b in table[(k, sd)]["corrected"].items()},
            }
            for k, sd in grid
        ],
        "selected": {"neutral_members": best[0], "state_sd": best[1]},
        "raw_argmin": {"neutral_members": best_raw[0], "state_sd": best_raw[1]},
        "seasonal_offsets": {
            "shipped": offsets[SHIPPED].tolist(),
            "selected": offsets[best].tolist(),
        },
        "transfer": transfer,
        "acceptance": accept,
        "band": band,
        "calibration_windows": [w["label"] for w in calib],
    }
    (out_dir / "dispersion_select_results.json").write_text(json.dumps(results, indent=1))
    print("\nseasonal offsets of the selected point (SWI, + = model too wet):")
    print("   " + " ".join(f"{m:>6s}" for m in MONTH_LABELS))
    print("   " + " ".join(f"{v:+6.3f}" for v in offsets[best]))
    print(f"\nwrote {spec_path} -- point config/inference_swigan.yaml model.dispersion at it")
    print(f"wrote {out_dir / 'dispersion_select_results.json'}")


if __name__ == "__main__":
    main()
