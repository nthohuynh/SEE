"""
PSO-based hyperparameter optimization for ensemble SEE models.

Uses mealpy library (dependency of mafese) for Particle Swarm Optimization
to optimize hyperparameters of ensemble regression models.

Example usage:
    ```python
    from ensemble_see.pso_optimizer import PSOConfig, PSOHyperparameterOptimizer
    from ensemble_see.models import build_bagging_lr
    import numpy as np

    # Prepare data (X_train, y_train should be scaled)
    X_train = ...  # shape: (n_samples, n_features)
    y_train = ...  # shape: (n_samples,)

    # Configure PSO
    pso_config = PSOConfig(
        n_particles=30,
        n_iterations=50,
        cv_folds=5,
        random_state=42,
        verbose=True,
    )

    # Create optimizer
    optimizer = PSOHyperparameterOptimizer(pso_config)

    # Optimize hyperparameters for a model
    best_params, best_mar = optimizer.optimize(
        model_name="B-LR",
        model_builder=build_bagging_lr,
        X_train=X_train,
        y_train=y_train,
    )

    print(f"Best hyperparameters: {best_params}")
    print(f"Best MAR: {best_mar:.6f}")
    ```

Integration with experiment:
    Set `use_pso=True` in `ExperimentConfig` or use `--use-pso` CLI flag.
    PSO will optimize ensemble models before running the Monte Carlo experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from mealpy import FloatVar, IntVar, PSO
from sklearn.base import clone
from sklearn.model_selection import cross_val_score

from ensemble_see.data import scale_fit_train_transform_test
from ensemble_see.metrics import mar


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
        """Decode normalized value [0, 1] to actual parameter value."""
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


def _set_bagging_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for BaggingRegressor models."""
    if "n_estimators" in params:
        model.set_params(n_estimators=params["n_estimators"])
    if "max_samples" in params:
        model.set_params(max_samples=params["max_samples"])
    if "max_features" in params:
        model.set_params(max_features=params["max_features"])
    # Set base estimator params if needed (for Ridge, Lasso, HuberRegressor)
    if "alpha" in params or "epsilon" in params:
        base_estimator = model.get_params()["estimator"]
        # Clone to avoid modifying the original
        from sklearn.base import clone
        base_estimator = clone(base_estimator)
        if hasattr(base_estimator, "set_params"):
            if "alpha" in params:
                base_estimator.set_params(alpha=params["alpha"])
            if "epsilon" in params:
                base_estimator.set_params(epsilon=params["epsilon"])
            model.set_params(estimator=base_estimator)
    return model


def _set_stacking_params(model: Any, params: dict[str, Any]) -> Any:
    """Set hyperparameters for StackingRegressor models."""
    if "cv" in params:
        model.set_params(cv=int(params["cv"]))
    # Note: base estimator params would require more complex handling
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


def _get_hyperparameter_configs() -> dict[str, ModelHyperparameterConfig]:
    """Get hyperparameter configurations (lazy import to avoid circular dependencies)."""
    from ensemble_see.models import (
        build_bagging_la,
        build_bagging_lr,
        build_bagging_ri,
        build_bagging_rr,
        build_stacking_la,
        build_stacking_lr,
        build_stacking_ri,
        build_stacking_rr,
    )

    return {
        "B-LR": ModelHyperparameterConfig(
            model_name="B-LR",
            build_function=build_bagging_lr,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators", "int", 10, 200),
                HyperparameterSpace("max_samples", "float", 0.5, 1.0),
                HyperparameterSpace("max_features", "float", 0.5, 1.0),
            ],
            param_setter=_set_bagging_params,
        ),
        "B-RR": ModelHyperparameterConfig(
            model_name="B-RR",
            build_function=build_bagging_rr,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators", "int", 10, 200),
                HyperparameterSpace("max_samples", "float", 0.5, 1.0),
                HyperparameterSpace("max_features", "float", 0.5, 1.0),
                HyperparameterSpace("alpha", "float", 1e-5, 1e-2, log_scale=True),
                HyperparameterSpace("epsilon", "float", 1.0, 2.0),
            ],
            param_setter=_set_bagging_params,
        ),
        "B-RI": ModelHyperparameterConfig(
            model_name="B-RI",
            build_function=build_bagging_ri,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators", "int", 10, 200),
                HyperparameterSpace("max_samples", "float", 0.5, 1.0),
                HyperparameterSpace("max_features", "float", 0.5, 1.0),
                HyperparameterSpace("alpha", "float", 0.1, 100.0, log_scale=True),
            ],
            param_setter=_set_bagging_params,
        ),
        "B-LA": ModelHyperparameterConfig(
            model_name="B-LA",
            build_function=build_bagging_la,
            hyperparameter_spaces=[
                HyperparameterSpace("n_estimators", "int", 10, 200),
                HyperparameterSpace("max_samples", "float", 0.5, 1.0),
                HyperparameterSpace("max_features", "float", 0.5, 1.0),
                HyperparameterSpace("alpha", "float", 1e-5, 1.0, log_scale=True),
            ],
            param_setter=_set_bagging_params,
        ),
        "ST-LR": ModelHyperparameterConfig(
            model_name="ST-LR",
            build_function=build_stacking_lr,
            hyperparameter_spaces=[
                HyperparameterSpace("cv", "int", 3, 10),
            ],
            param_setter=_set_stacking_params,
        ),
        "ST-RR": ModelHyperparameterConfig(
            model_name="ST-RR",
            build_function=build_stacking_rr,
            hyperparameter_spaces=[
                HyperparameterSpace("cv", "int", 3, 10),
            ],
            param_setter=_set_stacking_params,
        ),
        "ST-RI": ModelHyperparameterConfig(
            model_name="ST-RI",
            build_function=build_stacking_ri,
            hyperparameter_spaces=[
                HyperparameterSpace("cv", "int", 3, 10),
            ],
            param_setter=_set_stacking_params,
        ),
        "ST-LA": ModelHyperparameterConfig(
            model_name="ST-LA",
            build_function=build_stacking_la,
            hyperparameter_spaces=[
                HyperparameterSpace("cv", "int", 3, 10),
            ],
            param_setter=_set_stacking_params,
        ),
    }


@dataclass
class PSOConfig:
    """Configuration for PSO optimization."""

    n_particles: int = 30
    """Number of particles in the swarm."""
    n_iterations: int = 50
    """Number of PSO iterations."""
    c1: float = 2.05
    """Cognitive parameter."""
    c2: float = 2.05
    """Social parameter."""
    w_min: float = 0.4
    """Minimum inertia weight."""
    w_max: float = 0.9
    """Maximum inertia weight."""
    cv_folds: int = 5
    """Number of CV folds for fitness evaluation."""
    random_state: int | None = 42
    """Random state for reproducibility."""
    verbose: bool = True
    """Whether to print progress."""


class PSOHyperparameterOptimizer:
    """
    Particle Swarm Optimization-based hyperparameter optimizer for ensemble models.

    Uses mealpy library to optimize hyperparameters by minimizing MAR (Mean Absolute Residual)
    via cross-validation.
    """

    def __init__(self, config: PSOConfig) -> None:
        self.config = config

    def optimize(
        self,
        model_name: str,
        model_builder: Callable[[], Any] | None = None,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
    ) -> tuple[dict[str, Any], float]:
        """
        Optimize hyperparameters for a given model.

        Args:
            model_name: Name of the model (e.g., 'B-LR', 'ST-RI').
            model_builder: Function that creates a model instance. If None, uses default from config.
            X_train: Training features.
            y_train: Training targets.

        Returns:
            Tuple of (best_hyperparameters, best_mar_score).
        """
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must be provided")

        configs = _get_hyperparameter_configs()
        if model_name not in configs:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(configs.keys())}"
            )

        config = configs[model_name]
        if model_builder is None:
            model_builder = config.build_function
        n_params = len(config.hyperparameter_spaces)

        # Define bounds for mealpy (normalized to [0, 1])
        bounds = FloatVar(lb=[0.0] * n_params, ub=[1.0] * n_params)

        # Create objective function
        def objective_function(solution: np.ndarray) -> float:
            """Objective: minimize MAR via cross-validation."""
            # Decode solution to hyperparameters
            hyperparams = {}
            for i, space in enumerate(config.hyperparameter_spaces):
                hyperparams[space.name] = space.decode(solution[i])

            # Build model with hyperparameters
            model_template = model_builder()
            model = clone(model_template)
            model = config.param_setter(model, hyperparams)

            # Evaluate using cross-validation
            try:
                # Use negative MAR as score (sklearn maximizes, we minimize)
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=self.config.cv_folds,
                    scoring="neg_mean_absolute_error",
                    n_jobs=1,  # Sequential to avoid nested parallelism
                )
                mar_score = -np.mean(scores)  # Convert back to MAR (positive)
                return mar_score
            except Exception as e:
                # Return a large penalty for invalid configurations
                if self.config.verbose:
                    print(f"Warning: Invalid hyperparameters {hyperparams}: {e}")
                return 1e10

        # Define problem
        problem = {
            "bounds": bounds,
            "obj_func": objective_function,
            "minmax": "min",
            "name": f"Hyperparameter Optimization for {model_name}",
        }

        # Initialize PSO optimizer
        pso_model = PSO.OriginalPSO(
            epoch=self.config.n_iterations,
            pop_size=self.config.n_particles,
            c1=self.config.c1,
            c2=self.config.c2,
            w_min=self.config.w_min,
            w_max=self.config.w_max,
        )

        # Solve
        if self.config.verbose:
            print(f"Optimizing {model_name} with PSO...")
            print(f"  Particles: {self.config.n_particles}, Iterations: {self.config.n_iterations}")

        g_best = pso_model.solve(problem=problem)

        # Decode best solution to hyperparameters
        best_hyperparams = {}
        for i, space in enumerate(config.hyperparameter_spaces):
            best_hyperparams[space.name] = space.decode(g_best.solution[i])

        best_mar = float(g_best.target.fitness)

        if self.config.verbose:
            print(f"  Best MAR: {best_mar:.6f}")
            print(f"  Best hyperparameters: {best_hyperparams}")

        return best_hyperparams, best_mar

    def optimize_multiple_models(
        self,
        models: dict[str, Callable[[], Any]],
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> dict[str, tuple[dict[str, Any], float]]:
        """
        Optimize hyperparameters for multiple models.

        Args:
            models: Dictionary mapping model names to builder functions.
            X_train: Training features.
            y_train: Training targets.

        Returns:
            Dictionary mapping model names to (best_hyperparameters, best_mar_score).
        """
        results = {}
        for model_name, model_builder in models.items():
            try:
                hyperparams, mar_score = self.optimize(
                    model_name, model_builder, X_train, y_train
                )
                results[model_name] = (hyperparams, mar_score)
            except Exception as e:
                if self.config.verbose:
                    print(f"Error optimizing {model_name}: {e}")
                results[model_name] = ({}, float("inf"))
        return results
