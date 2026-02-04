"""
Data loading, normalization, and splitting for SEE experiments.

Uses Max-Min normalization (0–1) as per the reference plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler

from ensemble_see.config import DatasetConfig


class SEEFrame(NamedTuple):
    """Raw features and target (no scaling applied). Scaling is done per split in experiment."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def _load_arff(path: Path) -> pd.DataFrame:
    """Load ARFF file into a DataFrame (numeric columns only)."""
    data, meta = arff.loadarff(path)
    names = meta.names()
    # Convert structured array to DataFrame; decode bytes to str for object cols
    rows = []
    for row in data:
        r = []
        for i, name in enumerate(names):
            v = row[i]
            if isinstance(v, bytes):
                v = v.decode("utf-8").strip() if v else np.nan
            r.append(v)
        rows.append(r)
    df = pd.DataFrame(rows, columns=names)
    # Keep only numeric columns (ARFF numeric -> float)
    for c in names:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV or ARFF and return (features DataFrame, target Series).

    Drops configured columns and any row with missing values in target or features.
    """
    path = Path(config.path)
    if path.suffix.lower() == ".arff":
        df = _load_arff(path)
    else:
        df = pd.read_csv(path)
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


def scale_fit_train_transform_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit MinMaxScaler on training data only, then transform both train and test.

    Prevents data leakage: test set min/max never influence the scaling.
    Returns (X_train_s, y_train_s, X_test_s, y_test_s).
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    y_test = np.asarray(y_test, dtype=np.float64).ravel()
    scaler_X.fit(X_train)
    scaler_y.fit(y_train.reshape(-1, 1))
    X_train_s = scaler_X.transform(X_train)
    X_test_s = scaler_X.transform(X_test)
    y_train_s = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
    y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    return X_train_s, y_train_s, X_test_s, y_test_s


def prepare_see_frame(config: DatasetConfig) -> SEEFrame:
    """
    Load dataset and return raw (X, y). No scaling here; scaling is applied
    per split in the experiment (fit on train, transform train and test).
    """
    X_df, y_series = load_dataset(config)
    feature_names = list(X_df.columns)
    X = X_df.to_numpy().astype(np.float64)
    y = y_series.to_numpy().astype(np.float64)
    return SEEFrame(X=X, y=y, feature_names=feature_names)


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
