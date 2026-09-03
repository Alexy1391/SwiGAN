"""Module containing methods to compute some evaluation metrics."""

import numpy as np


def compute_rmse_per_date(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute spatial RMSE per predictions timestep."""
    n_elements = mask.sum()

    masked_y_pred = y_pred * mask
    masked_y_true = y_true * mask

    return np.sqrt(((masked_y_true - masked_y_pred) ** 2).sum(axis=(-2, -1)) / n_elements)


def compute_smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute mean symmetric absolute percentage error."""
    absolute_diff = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mape = absolute_diff / (denominator + 1e-8)
    masked_mape = mape * mask
    return 100 * masked_mape


def compute_smape_per_date(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute mean symmetric absolute percentage error."""
    n_elements = mask.sum()

    masked_mape = compute_smape(y_true, y_pred, mask)
    return masked_mape.sum(axis=(-2, -1)) / n_elements


def compute_mse_per_pixel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute the per-pixel MSE over a sequence of maps of shape (n_dates, H, W)."""
    mse = ((y_true - y_pred) ** 2).mean(axis=0)
    return np.where(mask, mse, np.nan)


def compute_rmse_per_pixel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute the per-pixel RMSE over a sequence of maps of shape (n_dates, H, W)."""
    return np.sqrt(compute_mse_per_pixel(y_true, y_pred, mask))


def compute_mae_per_pixel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute the per-pixel MAE over a sequence of maps of shape (n_dates, H, W)."""
    mae = np.abs(y_true - y_pred).mean(axis=0)
    return np.where(mask, mae, np.nan)


def compute_smape_per_pixel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute the per-pixel SMAPE over a sequence of maps of shape (n_dates, H, W)."""
    absolute_diff = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = (100 * absolute_diff / (denominator + 1e-8)).mean(axis=0)
    return np.where(mask, smape, np.nan)


def compute_r2_per_pixel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute the per-pixel coefficient of determination (R2) over a sequence of maps.

    Maps are of shape (n_dates, H, W).
    """
    y_true_mean = y_true.mean(axis=0, keepdims=True)
    residual_sum_squares = ((y_true - y_pred) ** 2).sum(axis=0)
    total_sum_squares = ((y_true - y_true_mean) ** 2).sum(axis=0)
    r2 = 1 - residual_sum_squares / (total_sum_squares + 1e-8)
    return np.where(mask, r2, np.nan)


def compute_correlation_per_pixel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Compute the per-pixel Pearson correlation coefficient over a sequence of maps.

    Maps are of shape (n_dates, H, W).
    """
    y_true_centered = y_true - y_true.mean(axis=0, keepdims=True)
    y_pred_centered = y_pred - y_pred.mean(axis=0, keepdims=True)

    covariance = (y_true_centered * y_pred_centered).sum(axis=0)
    std_product = np.sqrt((y_true_centered**2).sum(axis=0)) * np.sqrt(
        (y_pred_centered**2).sum(axis=0)
    )
    correlation = covariance / (std_product + 1e-8)
    return np.where(mask, correlation, np.nan)
