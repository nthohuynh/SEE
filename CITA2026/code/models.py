"""
Ensemble and baseline models for SEE.

Implements:
- Baseline: Linear Regression, ELM (2 and 5 hidden nodes).
- Bagging: B-LR, B-RR, B-RI, B-LA.
- Stacking: ST-LR, ST-RR, ST-RI, ST-LA.
- Boosting: GBR, XGB, LGBM, CAT, HGB.

Per plan: Robust Regression = HuberRegressor; meta/stacking uses CV on base outputs.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.linear_model import (
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.utils.validation import check_is_fitted

try:  # optional GPU backend for ELM
    import cupy as cp
    _HAS_CUPY = True
except Exception:  # pragma: no cover - GPU is optional
    cp = None
    _HAS_CUPY = False

try:  # optional dependency
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional package
    XGBRegressor = None  # type: ignore[assignment]
    _HAS_XGBOOST = False

try:  # optional dependency
    from lightgbm import LGBMRegressor
    _HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - optional package
    LGBMRegressor = None  # type: ignore[assignment]
    _HAS_LIGHTGBM = False

try:  # optional dependency
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
except Exception:  # pragma: no cover - optional package
    CatBoostRegressor = None  # type: ignore[assignment]
    _HAS_CATBOOST = False


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
        use_gpu: bool = True,
    ) -> None:
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.activation = activation
        self.random_state = random_state
        self.use_gpu = use_gpu

    def _activation(self, Z):
        if _HAS_CUPY:
            xp = cp.get_array_module(Z)
        else:
            xp = np
        if self.activation == "tanh":
            return xp.tanh(Z)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + xp.exp(-xp.clip(Z, -500, 500)))
        raise ValueError(f"Unknown activation: {self.activation}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> ELMRegressor:
        X_np = np.asarray(X, dtype=np.float64)
        y_np = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_np.shape

        use_gpu = bool(self.use_gpu and _HAS_CUPY)

        if use_gpu:
            rng = cp.random.default_rng(self.random_state)
            W = rng.standard_normal((n_features, self.n_hidden))
            b_vec = rng.standard_normal((self.n_hidden,))
            Xb = cp.asarray(X_np)
            H = self._activation(Xb @ W + b_vec)
            A = H.T @ H + self.alpha * cp.eye(self.n_hidden, dtype=cp.float64)
            rhs = H.T @ cp.asarray(y_np)
            beta = cp.linalg.solve(A, rhs)
            # Store weights on CPU for compatibility with sklearn cloning / pickling
            self.W_ = cp.asnumpy(W)
            self.b_ = cp.asnumpy(b_vec)
            self.beta_ = cp.asnumpy(beta)
        else:
            rng = np.random.default_rng(self.random_state)
            self.W_ = rng.standard_normal((n_features, self.n_hidden))
            self.b_ = rng.standard_normal((self.n_hidden,))
            H = self._activation(X_np @ self.W_ + self.b_)
            # Ridge solution: (H'H + alpha I)^{-1} H' y
            A = H.T @ H + self.alpha * np.eye(self.n_hidden)
            rhs = H.T @ y_np
            self.beta_ = np.linalg.solve(A, rhs)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["W_", "b_", "beta_"])
        X_np = np.asarray(X, dtype=np.float64)

        use_gpu = bool(self.use_gpu and _HAS_CUPY)
        if use_gpu:
            Xb = cp.asarray(X_np)
            W = cp.asarray(self.W_)
            b_vec = cp.asarray(self.b_)
            beta = cp.asarray(self.beta_)
            H = self._activation(Xb @ W + b_vec)
            y_pred = H @ beta
            return cp.asnumpy(y_pred)

        H = self._activation(X_np @ self.W_ + self.b_)
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
# Boosting models
# -----------------------------------------------------------------------------

def build_gradient_boosting(random_state: int | None = None) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=2,
        subsample=0.9,
        loss="squared_error",
        random_state=random_state,
    )


def build_hist_gradient_boosting(
    random_state: int | None = None,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=8,
        l2_regularization=1e-4,
        early_stopping=False,
        loss="squared_error",
        random_state=random_state,
    )


def build_xgboost(random_state: int | None = None) -> Any:
    if not _HAS_XGBOOST:
        raise ImportError("xgboost is not installed")
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=1,
        verbosity=0,
    )


def build_lightgbm(random_state: int | None = None) -> Any:
    if not _HAS_LIGHTGBM:
        raise ImportError("lightgbm is not installed")
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )


def build_catboost(random_state: int | None = None) -> Any:
    if not _HAS_CATBOOST:
        raise ImportError("catboost is not installed")
    return CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        loss_function="RMSE",
        random_state=random_state,
        allow_writing_files=False,
        verbose=False,
    )


# -----------------------------------------------------------------------------
# Registry: model_id -> builder
# -----------------------------------------------------------------------------

def get_ensemble_models(random_state: int | None = None) -> dict[str, Any]:
    """All ensemble models (bagging + stacking + boosting) as name -> model instance."""
    models = {
        # "GBR": build_gradient_boosting(random_state),
        # "HGB": build_hist_gradient_boosting(random_state),
        "B-LR": build_bagging_lr(random_state),
        "B-RR": build_bagging_rr(random_state),
        "B-RI": build_bagging_ri(random_state),
        "B-LA": build_bagging_la(random_state),
        "ST-LR": build_stacking_lr(random_state),
        "ST-RR": build_stacking_rr(random_state),
        "ST-RI": build_stacking_ri(random_state),
        "ST-LA": build_stacking_la(random_state),
    }
    if _HAS_XGBOOST:
        models["XGB"] = build_xgboost(random_state)
    else:
        warnings.warn("xgboost not installed; skipping XGB model", RuntimeWarning)
    if _HAS_LIGHTGBM:
        models["LGBM"] = build_lightgbm(random_state)
    else:
        warnings.warn("lightgbm not installed; skipping LGBM model", RuntimeWarning)
    if _HAS_CATBOOST:
        models["CAT"] = build_catboost(random_state)
    else:
        warnings.warn("catboost not installed; skipping CAT model", RuntimeWarning)
    return models


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
