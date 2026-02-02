"""
Ensemble and baseline models for SEE.

Implements:
- Baseline: Linear Regression, ELM (2 and 5 hidden nodes).
- Bagging: B-LR, B-RR, B-RI, B-LA.
- Stacking: ST-LR, ST-RR, ST-RI, ST-LA.

Per plan: Robust Regression = HuberRegressor; meta/stacking uses CV on base outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.linear_model import (
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.utils.validation import check_is_fitted


# -----------------------------------------------------------------------------
# ELM (Extreme Learning Machine) – baseline
# -----------------------------------------------------------------------------


class ELMRegressor(BaseEstimator, RegressorMixin):
    """
    Extreme Learning Machine: random hidden layer, output weights solved by least squares.

    Single hidden layer, fixed random weights; only output layer is trained.
    """

    def __init__(
        self,
        n_hidden: int = 10,
        alpha: float = 1e-5,
        activation: str = "tanh",
        random_state: int | None = None,
    ) -> None:
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.activation = activation
        self.random_state = random_state

    def _activation(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(Z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        raise ValueError(f"Unknown activation: {self.activation}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> ELMRegressor:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.standard_normal((n_features, self.n_hidden))
        self.b_ = rng.standard_normal((self.n_hidden,))
        H = self._activation(X @ self.W_ + self.b_)
        # Ridge solution: (H'H + alpha I)^{-1} H' y
        A = H.T @ H + self.alpha * np.eye(self.n_hidden)
        b = H.T @ y
        self.beta_ = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["W_", "b_", "beta_"])
        X = np.asarray(X, dtype=np.float64)
        H = self._activation(X @ self.W_ + self.b_)
        return H @ self.beta_


# -----------------------------------------------------------------------------
# Base estimators (shared by bagging and stacking)
# -----------------------------------------------------------------------------

def _linear() -> LinearRegression:
    return LinearRegression()


def _robust() -> HuberRegressor:
    return HuberRegressor(epsilon=1.35, max_iter=200, alpha=1e-4)


def _ridge() -> Ridge:
    return Ridge(alpha=1.0, random_state=42)


def _lasso() -> Lasso:
    return Lasso(alpha=1e-4, random_state=42)


# -----------------------------------------------------------------------------
# Bagging models: B-LR, B-RR, B-RI, B-LA
# -----------------------------------------------------------------------------

def build_bagging_lr(random_state: int | None = None) -> BaggingRegressor:
    return BaggingRegressor(
        estimator=_linear(),
        n_estimators=50,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        random_state=random_state,
    )


def build_bagging_rr(random_state: int | None = None) -> BaggingRegressor:
    return BaggingRegressor(
        estimator=_robust(),
        n_estimators=50,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        random_state=random_state,
    )


def build_bagging_ri(random_state: int | None = None) -> BaggingRegressor:
    return BaggingRegressor(
        estimator=_ridge(),
        n_estimators=50,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        random_state=random_state,
    )


def build_bagging_la(random_state: int | None = None) -> BaggingRegressor:
    return BaggingRegressor(
        estimator=_lasso(),
        n_estimators=50,
        max_samples=0.8,
        max_features=1.0,
        bootstrap=True,
        random_state=random_state,
    )


# -----------------------------------------------------------------------------
# Stacking models (plan: base learners + meta-predictor)
# ST-LR: base = Robust, Ridge, Lasso; meta = Linear
# ST-RR: base = Linear, Ridge, Lasso; meta = Robust
# ST-RI: base = Robust, Linear, Lasso; meta = Ridge
# ST-LA: base = Lasso, Linear, Ridge; meta = Lasso
# -----------------------------------------------------------------------------

def build_stacking_lr(random_state: int | None = None) -> StackingRegressor:
    return StackingRegressor(
        estimators=[
            ("robust", _robust()),
            ("ridge", _ridge()),
            ("lasso", _lasso()),
        ],
        final_estimator=_linear(),
        cv=5,
    )


def build_stacking_rr(random_state: int | None = None) -> StackingRegressor:
    return StackingRegressor(
        estimators=[
            ("linear", _linear()),
            ("ridge", _ridge()),
            ("lasso", _lasso()),
        ],
        final_estimator=_robust(),
        cv=5,
    )


def build_stacking_ri(random_state: int | None = None) -> StackingRegressor:
    return StackingRegressor(
        estimators=[
            ("robust", _robust()),
            ("linear", _linear()),
            ("lasso", _lasso()),
        ],
        final_estimator=_ridge(),
        cv=5,
    )


def build_stacking_la(random_state: int | None = None) -> StackingRegressor:
    return StackingRegressor(
        estimators=[
            ("lasso", _lasso()),
            ("linear", _linear()),
            ("ridge", _ridge()),
        ],
        final_estimator=_lasso(),
        cv=5,
    )


# -----------------------------------------------------------------------------
# Baseline models
# -----------------------------------------------------------------------------

def build_linear_baseline() -> LinearRegression:
    return _linear()


def build_elm_2(random_state: int | None = None) -> ELMRegressor:
    return ELMRegressor(n_hidden=2, alpha=1e-5, random_state=random_state)


def build_elm_5(random_state: int | None = None) -> ELMRegressor:
    return ELMRegressor(n_hidden=5, alpha=1e-5, random_state=random_state)


# -----------------------------------------------------------------------------
# Registry: model_id -> builder
# -----------------------------------------------------------------------------

def get_ensemble_models(random_state: int | None = None) -> dict[str, Any]:
    """All 8 ensemble models (bagging + stacking) as name -> model instance."""
    return {
        "B-LR": build_bagging_lr(random_state),
        "B-RR": build_bagging_rr(random_state),
        "B-RI": build_bagging_ri(random_state),
        "B-LA": build_bagging_la(random_state),
        "ST-LR": build_stacking_lr(random_state),
        "ST-RR": build_stacking_rr(random_state),
        "ST-RI": build_stacking_ri(random_state),
        "ST-LA": build_stacking_la(random_state),
    }


def get_baseline_models(random_state: int | None = None) -> dict[str, Any]:
    """Baseline models for comparison: Linear, ELM-2, ELM-5."""
    return {
        "Linear": build_linear_baseline(),
        "ELM-2": build_elm_2(random_state),
        "ELM-5": build_elm_5(random_state),
    }


def get_all_models(random_state: int | None = None) -> dict[str, Any]:
    """All models: baselines + ensemble."""
    return {
        **get_baseline_models(random_state),
        **get_ensemble_models(random_state),
    }
