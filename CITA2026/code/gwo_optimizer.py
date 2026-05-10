"""
GWO-based hyperparameter optimization for ensemble SEE models.

Uses mealpy library for Grey Wolf Optimizer (GWO) to optimize hyperparameters
of ensemble regression models.

Grey Wolf Optimizer (GWO) simulates the leadership hierarchy and hunting
mechanism of grey wolves in nature. The pack is divided into four ranks:
  - Alpha (α): best solution found so far
  - Beta  (β): second best solution
  - Delta (δ): third best solution
  - Omega (ω): remaining wolves — update positions guided by α, β, δ

Position update rule (per iteration):
    X_new = (X_α + X_β + X_δ) / 3

The parameter `a` decreases linearly from 2 → 0 over iterations, naturally
transitioning the swarm from exploration to exploitation without manual tuning
of inertia weights (unlike PSO).

Example usage:
    ```python
    from ensemble_see.gwo_optimizer import GWOConfig, GWOHyperparameterOptimizer
    from ensemble_see.models import build_bagging_lr
    import numpy as np

    # Prepare data (X_train, y_train should be scaled)
    X_train = ...  # shape: (n_samples, n_features)
    y_train = ...  # shape: (n_samples,)

    # Configure GWO
    gwo_config = GWOConfig(
        n_wolves=30,
        n_iterations=50,
        cv_folds=5,
        random_state=42,
        verbose=True,
    )

    # Create optimizer
    optimizer = GWOHyperparameterOptimizer(gwo_config)

    # Optimize hyperparameters for a model
    best_params, best_mse = optimizer.optimize(
        model_name="B-LR",
        model_builder=build_bagging_lr,
        X_train=X_train,
        y_train=y_train,
    )

    print(f"Best hyperparameters: {best_params}")
    print(f"Best CV MSE: {best_mse:.6f}")
    ```

Integration with experiment:
    Set `use_gwo=True` in `ExperimentConfig` or use `--use-gwo` CLI flag.
    GWO will optimize ensemble models before running the Monte Carlo experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Callable

import numpy as np
from mealpy import FloatVar, GWO
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from ensemble_see.data import scale_fit_train_transform_test


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

@dataclass
class HyperparameterSpace:
    """Defines the search space for hyperparameter optimization."""

    name: str
    """Name of the hyperparameter."""
    param_type: str
    """Type: 'float' or 'int'."""
    lower_bound: float
    """Lower bound of the search space."""
    upper_bound: float
    """Upper bound of the search space."""
    log_scale: bool = False
    """Whether to use log scale for this parameter."""

    def decode(self, value: float) -> float | int:
        """Decode a normalized value [0, 1] to an actual parameter value."""
        if self.log_scale:
            log_lb = np.log10(self.lower_bound)
            log_ub = np.log10(self.upper_bound)
            decoded = 10 ** (log_lb + value * (log_ub - log_lb))
        else:
            decoded = self.lower_bound + value * (self.upper_bound - self.lower_bound)

        if self.param_type == "int":
            return int(round(decoded))
        return float(decoded)


@dataclass
class ModelHyperparameterConfig:
    """Configuration for hyperparameter optimization of a specific model."""

    model_name: str
    """Name of the model (e.g., 'B-LR', 'ST-RI')."""
    build_function: Callable[[], Any]
    """Function that builds the model template."""
    hyperparameter_spaces: list[HyperparameterSpace]
    """List of hyperparameters to optimize."""
    param_setter: Callable[[Any, dict[str, Any]], Any]
    """Function that sets hyperparameters on a model instance."""


# ---------------------------------------------------------------------------
# Parameter setters — one per model family
# ---------------------------------------------------------------------------

def _set_bagging_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for BaggingRegressor models."""
    if "n_estimators" in params:
        model.set_params(n_estimators=params["n_estimators"])
    if "max_samples" in params:
        model.set_params(max_samples=params["max_samples"])
    if "max_features" in params:
        model.set_params(max_features=params["max_features"])
    # Set base estimator params if needed (Ridge, Lasso, HuberRegressor)
    if "alpha" in params or "epsilon" in params:
        base_estimator = model.get_params()["estimator"]
        base_estimator = clone(base_estimator)
        if hasattr(base_estimator, "set_params"):
            if "alpha" in params:
                base_estimator.set_params(alpha=params["alpha"])
            if "epsilon" in params:
                base_estimator.set_params(epsilon=params["epsilon"])
            model.set_params(estimator=base_estimator)
    return model


def _set_stacking_lr_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for ST-LR stacking model."""
    if "cv" in params:
        model.set_params(cv=int(params["cv"]))

    est_list = list(model.get_params()["estimators"])
    est_map = {name: est for name, est in est_list}

    if "alpha_robust" in params or "epsilon_robust" in params:
        robust = clone(est_map["robust"])
        if "alpha_robust" in params:
            robust.set_params(alpha=params["alpha_robust"])
        if "epsilon_robust" in params:
            robust.set_params(epsilon=params["epsilon_robust"])
        est_map["robust"] = robust

    if "alpha_ridge" in params:
        ridge = clone(est_map["ridge"])
        ridge.set_params(alpha=params["alpha_ridge"])
        est_map["ridge"] = ridge

    if "alpha_lasso" in params:
        lasso = clone(est_map["lasso"])
        lasso.set_params(alpha=params["alpha_lasso"])
        est_map["lasso"] = lasso

    model.set_params(estimators=[
        ("robust", est_map["robust"]),
        ("ridge",  est_map["ridge"]),
        ("lasso",  est_map["lasso"]),
    ])
    return model


def _set_stacking_rr_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for ST-RR stacking model."""
    if "cv" in params:
        model.set_params(cv=int(params["cv"]))

    est_list = list(model.get_params()["estimators"])
    est_map = {name: est for name, est in est_list}

    if "alpha_ridge_base" in params:
        ridge = clone(est_map["ridge"])
        ridge.set_params(alpha=params["alpha_ridge_base"])
        est_map["ridge"] = ridge

    if "alpha_lasso_base" in params:
        lasso = clone(est_map["lasso"])
        lasso.set_params(alpha=params["alpha_lasso_base"])
        est_map["lasso"] = lasso

    model.set_params(estimators=[
        ("linear", est_map["linear"]),
        ("ridge",  est_map["ridge"]),
        ("lasso",  est_map["lasso"]),
    ])

    final_est = clone(model.get_params()["final_estimator"])
    if "alpha_robust_final" in params:
        final_est.set_params(alpha=params["alpha_robust_final"])
    if "epsilon_robust_final" in params:
        final_est.set_params(epsilon=params["epsilon_robust_final"])
    model.set_params(final_estimator=final_est)
    return model


def _set_stacking_ri_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for ST-RI stacking model."""
    if "cv" in params:
        model.set_params(cv=int(params["cv"]))

    est_list = list(model.get_params()["estimators"])
    est_map = {name: est for name, est in est_list}

    if "alpha_robust" in params or "epsilon_robust" in params:
        robust = clone(est_map["robust"])
        if "alpha_robust" in params:
            robust.set_params(alpha=params["alpha_robust"])
        if "epsilon_robust" in params:
            robust.set_params(epsilon=params["epsilon_robust"])
        est_map["robust"] = robust

    if "alpha_lasso" in params:
        lasso = clone(est_map["lasso"])
        lasso.set_params(alpha=params["alpha_lasso"])
        est_map["lasso"] = lasso

    model.set_params(estimators=[
        ("robust", est_map["robust"]),
        ("linear", est_map["linear"]),
        ("lasso",  est_map["lasso"]),
    ])

    final_est = clone(model.get_params()["final_estimator"])
    if "alpha_ridge_final" in params:
        final_est.set_params(alpha=params["alpha_ridge_final"])
    model.set_params(final_estimator=final_est)
    return model


def _set_stacking_la_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for ST-LA stacking model."""
    if "cv" in params:
        model.set_params(cv=int(params["cv"]))

    est_list = list(model.get_params()["estimators"])
    est_map = {name: est for name, est in est_list}

    if "alpha_lasso_base" in params:
        lasso = clone(est_map["lasso"])
        lasso.set_params(alpha=params["alpha_lasso_base"])
        est_map["lasso"] = lasso

    if "alpha_ridge_base" in params:
        ridge = clone(est_map["ridge"])
        ridge.set_params(alpha=params["alpha_ridge_base"])
        est_map["ridge"] = ridge

    model.set_params(estimators=[
        ("lasso",  est_map["lasso"]),
        ("linear", est_map["linear"]),
        ("ridge",  est_map["ridge"]),
    ])

    final_est = clone(model.get_params()["final_estimator"])
    if "alpha_lasso_final" in params:
        final_est.set_params(alpha=params["alpha_lasso_final"])
    model.set_params(final_estimator=final_est)
    return model


def _set_ridge_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for Ridge estimator."""
    if "alpha" in params:
        model.set_params(alpha=params["alpha"])
    return model


def _set_lasso_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for Lasso estimator."""
    if "alpha" in params:
        model.set_params(alpha=params["alpha"])
    return model


def _set_huber_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for HuberRegressor."""
    if "alpha" in params:
        model.set_params(alpha=params["alpha"])
    if "epsilon" in params:
        model.set_params(epsilon=params["epsilon"])
    return model


def _set_gbr_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for GradientBoostingRegressor."""
    updates: dict[str, Any] = {}
    for name in ("n_estimators", "learning_rate", "max_depth", "min_samples_leaf", "subsample"):
        if name in params:
            updates[name] = params[name]
    if updates:
        model.set_params(**updates)
    return model


def _set_hgb_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for HistGradientBoostingRegressor."""
    updates: dict[str, Any] = {}
    for name in ("max_iter", "learning_rate", "max_depth", "min_samples_leaf", "l2_regularization"):
        if name in params:
            updates[name] = params[name]
    if updates:
        model.set_params(**updates)
    return model


def _set_xgb_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for XGBRegressor."""
    updates: dict[str, Any] = {}
    for name in (
        "n_estimators", "learning_rate", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree", "reg_lambda", "reg_alpha",
    ):
        if name in params:
            updates[name] = params[name]
    if updates:
        model.set_params(**updates)
    return model


def _set_lgbm_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for LGBMRegressor."""
    updates: dict[str, Any] = {}
    for name in (
        "n_estimators", "learning_rate", "num_leaves", "min_child_samples",
        "subsample", "colsample_bytree", "reg_lambda", "reg_alpha",
    ):
        if name in params:
            updates[name] = params[name]
    if updates:
        model.set_params(**updates)
    return model


def _set_catboost_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for CatBoostRegressor."""
    updates: dict[str, Any] = {}
    for name in ("iterations", "learning_rate", "depth", "l2_leaf_reg", "random_strength"):
        if name in params:
            updates[name] = params[name]
    if updates:
        model.set_params(**updates)
    return model


# ---------------------------------------------------------------------------
# Hyperparameter search space registry
# ---------------------------------------------------------------------------

def _get_hyperparameter_configs() -> dict[str, ModelHyperparameterConfig]:
    """Get hyperparameter configurations (lazy import to avoid circular dependencies)."""
    from ensemble_see.models import (
        build_bagging_la,
        build_bagging_lr,
        build_bagging_ri,
        build_bagging_rr,
        build_catboost,
        build_gradient_boosting,
        build_hist_gradient_boosting,
        build_lightgbm,
        build_stacking_la,
        build_stacking_lr,
        build_stacking_ri,
        build_stacking_rr,
        build_xgboost,
    )

    return {
        # --- Bagging models ---
        "B-LR": ModelHyperparameterConfig(
            model_name="B-LR",
            build_function=build_bagging_lr,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators", "int",   10,   200),
                HyperparameterSpace("max_samples",  "float", 0.5,  1.0),
                HyperparameterSpace("max_features", "float", 0.5,  1.0),
            ],
            param_setter=_set_bagging_params,
        ),
        "B-RR": ModelHyperparameterConfig(
            model_name="B-RR",
            build_function=build_bagging_rr,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators", "int",   10,   200),
                HyperparameterSpace("max_samples",  "float", 0.5,  1.0),
                HyperparameterSpace("max_features", "float", 0.5,  1.0),
                HyperparameterSpace("alpha",        "float", 1e-5, 1e-2, log_scale=True),
                HyperparameterSpace("epsilon",      "float", 1.0,  2.0),
            ],
            param_setter=_set_bagging_params,
        ),
        "B-RI": ModelHyperparameterConfig(
            model_name="B-RI",
            build_function=build_bagging_ri,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators", "int",   10,   200),
                HyperparameterSpace("max_samples",  "float", 0.5,  1.0),
                HyperparameterSpace("max_features", "float", 0.5,  1.0),
                HyperparameterSpace("alpha",        "float", 0.1,  100.0, log_scale=True),
            ],
            param_setter=_set_bagging_params,
        ),
        "B-LA": ModelHyperparameterConfig(
            model_name="B-LA",
            build_function=build_bagging_la,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators", "int",   10,  200),
                HyperparameterSpace("max_samples",  "float", 0.5, 1.0),
                HyperparameterSpace("max_features", "float", 0.5, 1.0),
                HyperparameterSpace("alpha",        "float", 1e-5, 1.0, log_scale=True),
            ],
            param_setter=_set_bagging_params,
        ),
        # --- Stacking models ---
        "ST-LR": ModelHyperparameterConfig(
            model_name="ST-LR",
            build_function=build_stacking_lr,
            hyperparameter_spaces=[
                HyperparameterSpace("cv",            "int",   3,    10),
                HyperparameterSpace("alpha_robust",  "float", 1e-5, 1e-2, log_scale=True),
                HyperparameterSpace("epsilon_robust","float", 1.0,  2.0),
                HyperparameterSpace("alpha_ridge",   "float", 0.1,  100.0, log_scale=True),
                HyperparameterSpace("alpha_lasso",   "float", 1e-5, 1.0,  log_scale=True),
            ],
            param_setter=_set_stacking_lr_params,
        ),
        "ST-RR": ModelHyperparameterConfig(
            model_name="ST-RR",
            build_function=build_stacking_rr,
            hyperparameter_spaces=[
                HyperparameterSpace("cv",                  "int",   3,    10),
                HyperparameterSpace("alpha_ridge_base",    "float", 0.1,  100.0, log_scale=True),
                HyperparameterSpace("alpha_lasso_base",    "float", 1e-5, 1.0,  log_scale=True),
                HyperparameterSpace("alpha_robust_final",  "float", 1e-5, 1e-2, log_scale=True),
                HyperparameterSpace("epsilon_robust_final","float", 1.0,  2.0),
            ],
            param_setter=_set_stacking_rr_params,
        ),
        "ST-RI": ModelHyperparameterConfig(
            model_name="ST-RI",
            build_function=build_stacking_ri,
            hyperparameter_spaces=[
                HyperparameterSpace("cv",              "int",   3,    10),
                HyperparameterSpace("alpha_robust",    "float", 1e-5, 1e-2, log_scale=True),
                HyperparameterSpace("epsilon_robust",  "float", 1.0,  2.0),
                HyperparameterSpace("alpha_lasso",     "float", 1e-5, 1.0,  log_scale=True),
                HyperparameterSpace("alpha_ridge_final","float",0.1,  100.0, log_scale=True),
            ],
            param_setter=_set_stacking_ri_params,
        ),
        "ST-LA": ModelHyperparameterConfig(
            model_name="ST-LA",
            build_function=build_stacking_la,
            hyperparameter_spaces=[
                HyperparameterSpace("cv",              "int",   3,    10),
                HyperparameterSpace("alpha_lasso_base","float", 1e-5, 1.0,  log_scale=True),
                HyperparameterSpace("alpha_ridge_base","float", 0.1,  100.0, log_scale=True),
                HyperparameterSpace("alpha_lasso_final","float",1e-5, 1.0,  log_scale=True),
            ],
            param_setter=_set_stacking_la_params,
        ),
        # --- Boosting models ---
        "GBR": ModelHyperparameterConfig(
            model_name="GBR",
            build_function=build_gradient_boosting,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators",    "int",   80,  500),
                HyperparameterSpace("learning_rate",   "float", 0.01, 0.20),
                HyperparameterSpace("max_depth",       "int",   2,   8),
                HyperparameterSpace("min_samples_leaf","int",   1,   20),
                HyperparameterSpace("subsample",       "float", 0.6, 1.0),
            ],
            param_setter=_set_gbr_params,
        ),
        "HGB": ModelHyperparameterConfig(
            model_name="HGB",
            build_function=build_hist_gradient_boosting,
            hyperparameter_spaces=[
                HyperparameterSpace("max_iter",          "int",   100, 600),
                HyperparameterSpace("learning_rate",     "float", 0.01, 0.20),
                HyperparameterSpace("max_depth",         "int",   3,   12),
                HyperparameterSpace("min_samples_leaf",  "int",   5,   40),
                HyperparameterSpace("l2_regularization", "float", 1e-6, 1e-1, log_scale=True),
            ],
            param_setter=_set_hgb_params,
        ),
        "XGB": ModelHyperparameterConfig(
            model_name="XGB",
            build_function=build_xgboost,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators",    "int",   100, 600),
                HyperparameterSpace("learning_rate",   "float", 0.01, 0.20),
                HyperparameterSpace("max_depth",       "int",   3,   10),
                HyperparameterSpace("min_child_weight","float", 0.5, 10.0),
                HyperparameterSpace("subsample",       "float", 0.6, 1.0),
                HyperparameterSpace("colsample_bytree","float", 0.6, 1.0),
                HyperparameterSpace("reg_lambda",      "float", 1e-3, 10.0, log_scale=True),
                HyperparameterSpace("reg_alpha",       "float", 1e-4, 10.0, log_scale=True),
            ],
            param_setter=_set_xgb_params,
        ),
        "LGBM": ModelHyperparameterConfig(
            model_name="LGBM",
            build_function=build_lightgbm,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators",      "int",   100, 600),
                HyperparameterSpace("learning_rate",     "float", 0.01, 0.20),
                HyperparameterSpace("num_leaves",        "int",   15,  127),
                HyperparameterSpace("min_child_samples", "int",   5,   60),
                HyperparameterSpace("subsample",         "float", 0.6, 1.0),
                HyperparameterSpace("colsample_bytree",  "float", 0.6, 1.0),
                HyperparameterSpace("reg_lambda",        "float", 1e-4, 10.0, log_scale=True),
                HyperparameterSpace("reg_alpha",         "float", 1e-4, 10.0, log_scale=True),
            ],
            param_setter=_set_lgbm_params,
        ),
        "CAT": ModelHyperparameterConfig(
            model_name="CAT",
            build_function=build_catboost,
            hyperparameter_spaces=[
                HyperparameterSpace("iterations",      "int",   100, 600),
                HyperparameterSpace("learning_rate",   "float", 0.01, 0.20),
                HyperparameterSpace("depth",           "int",   4,   10),
                HyperparameterSpace("l2_leaf_reg",     "float", 1.0, 20.0),
                HyperparameterSpace("random_strength", "float", 1e-3, 10.0, log_scale=True),
            ],
            param_setter=_set_catboost_params,
        ),
    }


# ---------------------------------------------------------------------------
# GWO configuration
# ---------------------------------------------------------------------------

@dataclass
class GWOConfig:
    """Configuration for GWO optimization.

    GWO requires no velocity/inertia parameters (unlike PSO).
    The exploration-exploitation balance is governed automatically
    by the linearly decreasing parameter `a` (2 → 0).
    """

    n_wolves: int = 30
    """Number of wolves in the pack (population size)."""
    n_iterations: int = 50
    """Number of GWO iterations (hunting epochs)."""
    cv_folds: int = 5
    """Number of CV folds for fitness evaluation."""
    random_state: int | None = 42
    """Random state for reproducibility."""
    verbose: bool = True
    """Whether to print progress."""
    convergence_plot_dir: Path | None = None
    """If set, save convergence curve (fitness vs iteration) to this directory."""


# ---------------------------------------------------------------------------
# GWO optimizer
# ---------------------------------------------------------------------------

class GWOHyperparameterOptimizer:
    """
    Grey Wolf Optimizer-based hyperparameter optimizer for ensemble SEE models.

    Simulates the social hierarchy (α, β, δ, ω) and cooperative hunting of
    grey wolves to minimize cross-validation MSE over the hyperparameter space.

    Key differences from PSO:
    - No velocity, cognitive (c1) or social (c2) parameters required.
    - Position update driven by the three best solutions (α, β, δ).
    - Parameter `a` decreases 2 → 0 automatically, balancing exploration vs
      exploitation without manual tuning.
    """

    def __init__(self, config: GWOConfig) -> None:
        self.config = config

    def _sanitize_hyperparams(self, hyperparams: dict[str, Any], n_samples: int) -> dict[str, Any]:
        """Clamp decoded hyperparameters that depend on data size."""
        sanitized = dict(hyperparams)
        if "cv" in sanitized:
            outer_cv = max(2, int(self.config.cv_folds))
            min_outer_train = n_samples - math.ceil(n_samples / outer_cv)
            max_inner_cv = max(2, min_outer_train)
            sanitized["cv"] = max(2, min(int(sanitized["cv"]), max_inner_cv))
        return sanitized

    def optimize(
        self,
        model_name: str,
        model_builder: Callable[[], Any] | None = None,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        *,
        dataset_name: str | None = None,
    ) -> tuple[dict[str, Any], float]:
        """
        Optimize hyperparameters for a given model using GWO.

        Args:
            model_name: Name of the model (e.g., 'B-LR', 'ST-RI').
            model_builder: Function that creates a model instance.
                           If None, uses the default from the config registry.
            X_train: Training features.
            y_train: Training targets.
            dataset_name: Optional dataset name used in convergence plot filenames.

        Returns:
            Tuple of (best_hyperparameters, best_fitness).
            Fitness is CV MSE (minimized); lower is better.
        """
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided.")

        configs = _get_hyperparameter_configs()
        if model_name not in configs:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(configs.keys())}"
            )

        model_cfg = configs[model_name]
        if model_builder is None:
            model_builder = model_cfg.build_function
        n_params = len(model_cfg.hyperparameter_spaces)

        # GWO searches in [0, 1]^n; each dimension is later decoded by HyperparameterSpace
        bounds = FloatVar(lb=[0.0] * n_params, ub=[1.0] * n_params)

        # ------------------------------------------------------------------
        # Objective function: decode wolf position → hyperparams → CV MSE
        # ------------------------------------------------------------------
        def objective_function(solution: np.ndarray) -> float:
            hyperparams: dict[str, Any] = {}
            for i, space in enumerate(model_cfg.hyperparameter_spaces):
                hyperparams[space.name] = space.decode(solution[i])
            hyperparams = self._sanitize_hyperparams(hyperparams, len(y_train))

            model_template = model_builder()
            model = clone(model_template)
            model = model_cfg.param_setter(model, hyperparams)

            try:
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=self.config.cv_folds,
                    scoring="neg_mean_squared_error",
                    n_jobs=1,  # Sequential to avoid nested parallelism
                )
                return float(-np.mean(scores))  # Convert neg-MSE → MSE (positive)
            except Exception as exc:
                if self.config.verbose:
                    print(f"  [GWO] Warning: invalid hyperparams {hyperparams}: {exc}")
                return 1e10  # Large penalty for invalid configurations

        # ------------------------------------------------------------------
        # Define mealpy problem dict
        # ------------------------------------------------------------------
        problem = {
            "bounds": bounds,
            "obj_func": objective_function,
            "minmax": "min",
            "name": f"GWO Hyperparameter Optimization — {model_name}",
        }

        # ------------------------------------------------------------------
        # Initialise GWO solver
        # ------------------------------------------------------------------
        gwo_model = GWO.OriginalGWO(
            epoch=self.config.n_iterations,
            pop_size=self.config.n_wolves,
        )

        if self.config.verbose:
            print(f"\n[GWO] Optimizing {model_name}...")
            print(f"      Wolves: {self.config.n_wolves} | Iterations: {self.config.n_iterations}")
            print(f"      Search dims: {n_params} | CV folds: {self.config.cv_folds}")

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        g_best = gwo_model.solve(problem=problem)

        # ------------------------------------------------------------------
        # Extract convergence history
        # ------------------------------------------------------------------
        hist = getattr(gwo_model, "history", None)
        if hist is not None and getattr(hist, "list_global_best_fit", None):
            list_g = hist.list_global_best_fit
            best_fitness = float(min(list_g))
            best_epoch   = int(np.argmin(list_g))
            n_epochs     = len(list_g)

            if self.config.verbose:
                print(
                    f"  [GWO] Convergence: best fitness (CV MSE) = {best_fitness:.6f} "
                    f"at iteration {best_epoch} (of 0..{n_epochs - 1})"
                )
                # Sample convergence table
                step = max(1, n_epochs // 10)
                indices = list(range(0, n_epochs, step))
                if indices[-1] != n_epochs - 1:
                    indices.append(n_epochs - 1)
                print("  Epoch | Global best α fitness (CV MSE)")
                print("  ------+---------------------------------")
                for i in indices:
                    marker = " ← best" if i == best_epoch else ""
                    print(f"  {i:5} | {list_g[i]:.6f}{marker}")

            if self.config.convergence_plot_dir is not None:
                self._save_convergence_plot(
                    list_g,
                    model_name=model_name,
                    dataset_name=dataset_name or "dataset",
                    best_epoch=best_epoch,
                    best_fitness=best_fitness,
                )
        else:
            best_fitness = float(g_best.target.fitness)
            if self.config.verbose:
                print(f"  [GWO] Best α fitness (CV MSE): {best_fitness:.6f}")

        # ------------------------------------------------------------------
        # Decode alpha wolf position → best hyperparameters
        # ------------------------------------------------------------------
        best_hyperparams: dict[str, Any] = {}
        for i, space in enumerate(model_cfg.hyperparameter_spaces):
            best_hyperparams[space.name] = space.decode(g_best.solution[i])
        best_hyperparams = self._sanitize_hyperparams(best_hyperparams, len(y_train))

        if self.config.verbose:
            print(f"  [GWO] Best hyperparameters (α): {best_hyperparams}\n")

        return best_hyperparams, best_fitness

    # ------------------------------------------------------------------
    # Convergence plot
    # ------------------------------------------------------------------
    def _save_convergence_plot(
        self,
        list_global_best_fit: list[float],
        model_name: str,
        dataset_name: str,
        best_epoch: int,
        best_fitness: float,
    ) -> None:
        """Save the GWO convergence curve (α fitness vs iteration) as a PNG."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        out_dir = Path(self.config.convergence_plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        epochs = range(len(list_global_best_fit))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            epochs, list_global_best_fit,
            color="steelblue", linewidth=1.5,
            label="Alpha wolf (CV MSE)",
        )
        ax.axvline(
            x=best_epoch, color="coral", linestyle="--", alpha=0.8,
            label=f"Best at iter {best_epoch}",
        )
        ax.scatter([best_epoch], [best_fitness], color="red", s=60, zorder=5)
        ax.set_xlabel("Iteration (epoch)")
        ax.set_ylabel("Alpha fitness (CV MSE, minimized)")
        ax.set_title(f"GWO convergence — {dataset_name} — {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = f"{dataset_name}_{model_name}_gwo_convergence.png"
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()

        if self.config.verbose:
            print(f"  [GWO] Convergence plot saved: {out_dir / filename}")

    # ------------------------------------------------------------------
    # Batch optimization
    # ------------------------------------------------------------------
    def optimize_multiple_models(
        self,
        models: dict[str, Callable[[], Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        dataset_name: str | None = None,
    ) -> dict[str, tuple[dict[str, Any], float]]:
        """
        Optimize hyperparameters for multiple models sequentially.

        Args:
            models: Mapping of model_name → builder function.
            X_train: Training features.
            y_train: Training targets.
            dataset_name: Optional dataset name for convergence plot filenames.

        Returns:
            Mapping of model_name → (best_hyperparameters, best_fitness).
            Models that fail are stored with ({}, inf).
        """
        results: dict[str, tuple[dict[str, Any], float]] = {}
        for model_name, model_builder in models.items():
            try:
                hyperparams, fitness = self.optimize(
                    model_name, model_builder, X_train, y_train,
                    dataset_name=dataset_name,
                )
                results[model_name] = (hyperparams, fitness)
            except Exception as exc:
                if self.config.verbose:
                    print(f"  [GWO] Error optimizing {model_name}: {exc}")
                results[model_name] = ({}, float("inf"))
        return results
