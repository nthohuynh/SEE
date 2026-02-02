"""
Evaluation metrics and statistical tests for SEE.

- MAR/MAE: Mean Absolute Residual (primary performance index).
- MSE, RMSE, R²: regression metrics (sklearn.metrics).
- RG: Relative Gain vs baseline.
- Kolmogorov-Smirnov (normality), Wilcoxon (significance).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def mar(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Residual:
        MAR = (1/n) * sum(|y_i - y_hat_i|)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    return float(np.mean(np.abs(y_true - y_pred)))


def relative_gain(error_baseline: float, error_proposed: float) -> float:
    """
    Relative Gain (percentage):
        RG = 100 * (Error_a - Error_b) / Error_a

    Positive RG means the proposed model has lower error than baseline.
    """
    if error_baseline <= 0:
        return 0.0
    return 100.0 * (error_baseline - error_proposed) / error_baseline


def kolmogorov_smirnov_test(residuals: np.ndarray) -> tuple[float, float]:
    """
    Test normality of residuals. Returns (statistic, p-value).
    """
    residuals = np.asarray(residuals, dtype=np.float64).ravel()
    stat, pval = stats.kstest(residuals, "norm", args=(np.mean(residuals), np.std(residuals)))
    return float(stat), float(pval)


def wilcoxon_signed_rank(
    errors_a: np.ndarray, errors_b: np.ndarray, alternative: str = "less"
) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test: compare two paired error distributions.

    alternative='less': H1 that errors in second group (b) are smaller.
    Returns (statistic, p-value).
    """
    errors_a = np.asarray(errors_a, dtype=np.float64).ravel()
    errors_b = np.asarray(errors_b, dtype=np.float64).ravel()
    if len(errors_a) != len(errors_b):
        raise ValueError("errors_a and errors_b must have same length")
    stat, pval = stats.wilcoxon(errors_a, errors_b, alternative=alternative)
    return float(stat), float(pval)


class MARResult(NamedTuple):
    """MAR and optional per-sample absolute residuals for statistical tests."""

    mar: float
    abs_residuals: np.ndarray


def mar_with_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> MARResult:
    """Compute MAR and return absolute residuals for downstream tests."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    abs_res = np.abs(y_true - y_pred)
    return MARResult(mar=float(np.mean(abs_res)), abs_residuals=abs_res)


# -----------------------------------------------------------------------------
# Regression metrics (MSE, RMSE, MAE, R²) via sklearn
# -----------------------------------------------------------------------------

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(mean_squared_error(y_true, y_pred))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (same units as target)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (equivalent to MAR)."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² (coefficient of determination). Can be negative for poor fits."""
    return float(r2_score(y_true, y_pred))


def compute_all_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Return dict with MAR, MSE, RMSE, MAE, R² for y_true vs y_pred."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return {
        "MAR": mar(y_true, y_pred),
        "MSE": compute_mse(y_true, y_pred),
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "R2": compute_r2(y_true, y_pred),
    }
