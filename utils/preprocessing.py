"""Module containing preprocessing methods."""

from collections.abc import Mapping

import numpy as np
import pandas as pd


def coerce_comma_decimal_columns(input_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce numeric columns stored as comma-decimal strings (e.g. "0,564") to floats."""
    for col in columns:
        if input_df[col].dtype == object:
            input_df[col] = pd.to_numeric(
                input_df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce"
            )
    return input_df


def dataframe_to_rasters(
    input_df: pd.DataFrame,
    target_cols: str,
    input_cols: list[str],
    height: int,
    width: int,
    x_dim_col: str = "x",
    y_dim_col: str = "y",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract the maps and the timestamps from the input dataset."""
    months = sorted(input_df["month"].unique())
    years = sorted(input_df["year"].unique())
    input_df = input_df.sort_values(["year", "month", y_dim_col, x_dim_col])

    input_maps, target_maps, timestamps = [], [], []
    mask = None

    input_df.sort_values(["year", "month", y_dim_col, x_dim_col], inplace=True)

    for y in years:
        for m in months:
            sub_df = input_df[(input_df["year"] == y) & (input_df["month"] == m)]
            if len(sub_df) != height * width:
                continue  # Skip incomplete months

            x_i = np.stack(
                [sub_df[var].values.reshape(height, width) for var in input_cols], axis=0
            )
            y_i = sub_df[target_cols].values.reshape(1, height, width)

            t_vec = [m - 1]

            input_maps.append(x_i.astype(np.float32))
            target_maps.append(y_i.astype(np.float32))
            timestamps.append(t_vec)

            if mask is None:
                mask = sub_df["mask"].values.reshape(1, height, width)

    input_maps = np.stack(input_maps, axis=0)
    target_maps = np.stack(target_maps, axis=0)
    timestamps = np.stack(timestamps, axis=0).astype(np.float32)

    return input_maps, target_maps, timestamps, mask


def single_channel_statistic(statistics: Mapping[str, np.ndarray], name: str) -> float:
    """Read one standardization statistic as a scalar.

    They are saved with ``keepdims=True``, i.e. shaped ``(1, 1, 1, 1)``. The target has a single
    channel, so the value is a scalar; used at its saved shape it would broadcast a
    ``(T, H, W)`` stack of maps up to 4-D and silently change what every consumer downstream
    stacks and reduces over.
    """
    value = np.asarray(statistics[name], dtype=np.float32).reshape(-1)
    if value.size != 1:
        raise ValueError(
            f"'{name}' holds {value.size} values, so the target is not single-channel and this "
            "standardization would be wrong. The SwiGAN generator assumes output_channels=1."
        )
    return float(value[0])


def standardize_swi_history(
    history_maps: np.ndarray,
    statistics: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> np.ndarray:
    """Put raw SWI frames into the standardized space the generator's history channels expect.

    WHY THIS EXISTS. The generator's last ``num_input_steps`` input channels are SWI frames, and
    training feeds them STANDARDIZED: ``utils/swi_dataset.py`` applies
    ``np.where(mask, (targets - targets_mean) / targets_std, 0.0)`` to every split before a
    ``SWIDataset`` ever sees it. ``dataframe_to_rasters`` returns RAW SWI, so an inference path
    that seeds its rollout straight from a dataframe hands the model a unit system it was never
    fitted on -- and only for the first ``num_input_steps`` steps, because from then on the
    buffer holds the generator's own output, which is already standardized.

    On Corse (``targets_mean`` 0.536, ``targets_std`` 0.385) that mistake fed the history a
    +0.90 sigma offset at 0.385x contrast: a bone-dry August of -0.03 reached the generator as
    0.52, i.e. as average soil.

    ORDER MATTERS, and it matches the trainer: standardize first, THEN zero the padding, so a
    masked-out cell reads as 0 -- the training mean -- and not as
    ``(0 - targets_mean) / targets_std``.
    """
    mean = single_channel_statistic(statistics, "targets_mean")
    std = single_channel_statistic(statistics, "targets_std")
    return np.where(np.squeeze(mask).astype(bool), (history_maps - mean) / std, 0.0)


def fill_all_missing_pixels(
    input_df: pd.DataFrame, y_dim_col: str = "y", x_dim_col: str = "x", fill_value: float = 0.0
) -> pd.DataFrame:
    """Fill missing (x, y) combinations for all (year, month) in the dataframe.

    Adds rows with `fill_value` for all columns not in ['x', 'y', 'year', 'month'].
    """
    # Get unique x and y positions
    unique_x = input_df["x"].unique()
    unique_y = input_df["y"].unique()

    # Create full grid of x and y
    full_grid = pd.MultiIndex.from_product(
        [unique_x, unique_y], names=[x_dim_col, y_dim_col]
    ).to_frame(index=False)

    # Get all year/month combinations
    unique_dates = input_df[["year", "month"]].drop_duplicates()

    # Get other columns (to fill with default value)
    value_cols = [
        col for col in input_df.columns if col not in [x_dim_col, y_dim_col, "year", "month"]
    ]

    # Accumulator for new rows
    filled_parts = []

    for _, row in unique_dates.iterrows():
        yr, mo = row["year"], row["month"]

        # Filter original data
        df_subset = input_df[(input_df["year"] == yr) & (input_df["month"] == mo)]

        # Merge with full grid to find missing positions
        merged = full_grid.merge(df_subset, on=[x_dim_col, y_dim_col], how="left", indicator=True)
        missing = merged[merged["_merge"] == "left_only"][[x_dim_col, y_dim_col]]

        if not missing.empty:
            missing["year"] = yr
            missing["month"] = mo
            for col in value_cols:
                missing[col] = fill_value
            filled_parts.append(missing)

    # Combine original data with all filled parts
    if filled_parts:
        df_filled = pd.concat([input_df] + filled_parts, ignore_index=True)
    else:
        df_filled = input_df.copy()

    # Optional: sort and reset index
    return df_filled.sort_values(by=["year", "month", y_dim_col, x_dim_col]).reset_index(drop=True)
