"""Module containing preprocessing methods."""

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
