"""
Data loading, normalization, and splitting for SEE experiments.

Uses Max-Min normalization (0–1) as per the reference plan.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ensemble_see.config import DatasetConfig


class SEEFrame(NamedTuple):
    """Features and target with optional scaler for inverse transform."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler


def load_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV and return (features DataFrame, target Series).

    Drops configured columns and any row with missing values in target or features.
    """
    df = pd.read_csv(config.path)
    if config.target_column not in df.columns:
        raise ValueError(
            f"Target column '{config.target_column}' not in {list(df.columns)}"
        )
    drop = [c for c in config.drop_columns if c in df.columns]
    if drop:
        df = df.drop(columns=drop)
    cols = [c for c in df.columns if c != config.target_column]
    # Keep only numeric feature columns
    numeric = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in cols if c in numeric.columns]
    if not feature_cols:
        raise ValueError(f"No numeric feature columns found; columns: {list(df.columns)}")
    X = df[feature_cols]
    y = df[config.target_column]
    # Drop rows with any NaN in X or y
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].astype(np.float64)
    y = y.loc[mask].astype(np.float64)
    return X, y


def normalize_max_min(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Normalize X and y to [0, 1] using Max-Min (MinMaxScaler).

    Returns (X_scaled, y_scaled, scaler_X, scaler_y).
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(np.asarray(X, dtype=np.float64))
    y_1d = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(y_1d).ravel()
    return X_scaled, y_scaled, scaler_X, scaler_y


def prepare_see_frame(config: DatasetConfig) -> SEEFrame:
    """
    Load dataset, normalize to [0,1], and return SEEFrame.
    """
    X_df, y_series = load_dataset(config)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy()
    y = y_series.to_numpy()
    X_scaled, y_scaled, scaler_X, scaler_y = normalize_max_min(X, y)
    return SEEFrame(
        X=X_scaled,
        y=y_scaled,
        feature_names=feature_names,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
    )


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle and split into train/test by ratio.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    n_train = int(n * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, X_test, y_train, y_test
