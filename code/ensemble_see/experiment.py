"""
Monte Carlo experiment loop for SEE ensemble study.

Follows plan/ensemble_regression_see.md (Carvalho et al., IJSEA 2020):
- Shuffle → Split 70% train / 30% test → Train on train only → Predict on test → MAR.
- Normalization: Max-Min [0,1] fit on training data only, then transform train and test
  (no leakage). StackingRegressor uses cv=5 for meta-learner (out-of-fold base predictions).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone

from ensemble_see.config import DatasetConfig, ExperimentConfig
from ensemble_see.data import (
    prepare_see_frame,
    scale_fit_train_transform_test,
    train_test_split,
)
from ensemble_see.metrics import (
    compute_all_regression_metrics,
    mar_with_residuals,
    wilcoxon_signed_rank,
)
from ensemble_see.models import get_baseline_models, get_ensemble_models
from ensemble_see.pso_optimizer import PSOConfig, PSOHyperparameterOptimizer


@dataclass
class IterationResult:
    """MAR and regression metrics per model for one train/test split."""

    mar_by_model: dict[str, float]
    metrics_by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    abs_residuals_by_model: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Aggregated results over all iterations."""

    dataset_name: str
    n_iterations: int
    n_train: int
    n_test: int
    mar_mean: dict[str, float]
    mar_std: dict[str, float]
    mse_mean: dict[str, float] = field(default_factory=dict)
    mse_std: dict[str, float] = field(default_factory=dict)
    rmse_mean: dict[str, float] = field(default_factory=dict)
    rmse_std: dict[str, float] = field(default_factory=dict)
    mae_mean: dict[str, float] = field(default_factory=dict)
    mae_std: dict[str, float] = field(default_factory=dict)
    r2_mean: dict[str, float] = field(default_factory=dict)
    r2_std: dict[str, float] = field(default_factory=dict)
    relative_gain_vs_linear: dict[str, float] = field(default_factory=dict)
    wilcoxon_vs_linear: dict[str, float] = field(default_factory=dict)
    """p-value of Wilcoxon test (errors_linear vs errors_model); 'less' alternative."""


def _run_single_iteration(
    X: np.ndarray,
    y: np.ndarray,
    models: dict[str, Any],
    config: ExperimentConfig,
    iter_seed: int,
) -> IterationResult:
    # 1. Shuffle and split (plan: 70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_ratio=config.train_ratio,
        random_state=iter_seed,
    )
    # 2. Fit scalers on train only; transform train and test (no leakage)
    X_train_s, y_train_s, X_test_s, y_test_s = scale_fit_train_transform_test(
        X_train, y_train, X_test, y_test
    )
    mar_by_model: dict[str, float] = {}
    metrics_by_model: dict[str, dict[str, float]] = {}
    abs_residuals_by_model: dict[str, np.ndarray] = {}

    for name, template in models.items():
        model = clone(template)
        try:
            model.set_params(random_state=iter_seed)
        except (TypeError, ValueError):
            pass
        model.fit(X_train_s, y_train_s)
        y_pred = model.predict(X_test_s)
        res = mar_with_residuals(y_test_s, y_pred)
        mar_by_model[name] = res.mar
        abs_residuals_by_model[name] = res.abs_residuals
        metrics_by_model[name] = compute_all_regression_metrics(y_test_s, y_pred)

    return IterationResult(
        mar_by_model=mar_by_model,
        metrics_by_model=metrics_by_model,
        abs_residuals_by_model=abs_residuals_by_model,
    )


def run_experiment(
    dataset_config: DatasetConfig,
    experiment_config: ExperimentConfig,
    *,
    include_baselines: bool = True,
    include_ensemble: bool = True,
) -> ExperimentResult:
    """
    Run the full Monte Carlo experiment and return aggregated MAR (mean ± std)
    and optional relative gain / Wilcoxon vs Linear baseline.
    """
    frame = prepare_see_frame(dataset_config)
    X, y = frame.X, frame.y
    n = len(y)
    n_train = int(n * experiment_config.train_ratio)
    n_test = n - n_train

    models: dict[str, Any] = {}
    if include_baselines:
        models.update(get_baseline_models(experiment_config.random_state))
    if include_ensemble:
        models.update(get_ensemble_models(experiment_config.random_state))
    if not models:
        raise ValueError("At least one of include_baselines or include_ensemble must be True")

    # PSO hyperparameter optimization (if enabled)
    optimized_hyperparams: dict[str, dict[str, Any]] = {}
    if experiment_config.use_pso and include_ensemble:
        # Use a single train/test split for PSO optimization
        rng = np.random.default_rng(experiment_config.random_state)
        perm = rng.permutation(n)
        X_shuf, y_shuf = X[perm], y[perm]
        X_train_pso, X_test_pso, y_train_pso, y_test_pso = train_test_split(
            X_shuf, y_shuf,
            train_ratio=experiment_config.train_ratio,
            random_state=experiment_config.random_state,
        )
        X_train_pso_s, y_train_pso_s, _, _ = scale_fit_train_transform_test(
            X_train_pso, y_train_pso, X_test_pso, y_test_pso
        )

        pso_config = PSOConfig(
            n_particles=experiment_config.pso_n_particles,
            n_iterations=experiment_config.pso_n_iterations,
            cv_folds=experiment_config.pso_cv_folds,
            random_state=experiment_config.random_state,
            verbose=True,
        )
        optimizer = PSOHyperparameterOptimizer(pso_config)

        # Get builder functions for ensemble models
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
        builder_map = {
            "B-LR": build_bagging_lr,
            "B-RR": build_bagging_rr,
            "B-RI": build_bagging_ri,
            "B-LA": build_bagging_la,
            "ST-LR": build_stacking_lr,
            "ST-RR": build_stacking_rr,
            "ST-RI": build_stacking_ri,
            "ST-LA": build_stacking_la,
        }
        ensemble_models_builders = {
            name: builder_map[name]
            for name in models.keys()
            if name in builder_map
        }

        if ensemble_models_builders:
            pso_results = optimizer.optimize_multiple_models(
                ensemble_models_builders, X_train_pso_s, y_train_pso_s
            )
            optimized_hyperparams = {
                name: params for name, (params, _) in pso_results.items()
            }
            # Update models with optimized hyperparameters
            from ensemble_see.pso_optimizer import _get_hyperparameter_configs
            configs = _get_hyperparameter_configs()
            for name, hyperparams in optimized_hyperparams.items():
                if name in models and hyperparams and name in configs:
                    # Create new model with optimized hyperparameters
                    base_model = models[name]
                    optimized_model = clone(base_model)
                    models[name] = configs[name].param_setter(optimized_model, hyperparams)

    n_iters = experiment_config.effective_iterations()
    rng = np.random.default_rng(experiment_config.random_state)
    iteration_seeds = rng.integers(0, 2**31, size=n_iters)

    all_mar: dict[str, list[float]] = {name: [] for name in models}
    all_mse: dict[str, list[float]] = {name: [] for name in models}
    all_rmse: dict[str, list[float]] = {name: [] for name in models}
    all_mae: dict[str, list[float]] = {name: [] for name in models}
    all_r2: dict[str, list[float]] = {name: [] for name in models}
    all_abs_residuals: dict[str, list[np.ndarray]] = {name: [] for name in models}

    for i in range(n_iters):
        # Use a different shuffle each iteration; pass seed for reproducibility
        X_shuf, y_shuf = X.copy(), y.copy()
        perm = np.random.default_rng(iteration_seeds[i]).permutation(n)
        X_shuf, y_shuf = X_shuf[perm], y_shuf[perm]
        res = _run_single_iteration(
            X_shuf, y_shuf, models, experiment_config, int(iteration_seeds[i])
        )
        for name, m in res.mar_by_model.items():
            all_mar[name].append(m)
        for name, arr in res.abs_residuals_by_model.items():
            all_abs_residuals[name].append(arr)
        for name, metrics in res.metrics_by_model.items():
            all_mse[name].append(metrics["MSE"])
            all_rmse[name].append(metrics["RMSE"])
            all_mae[name].append(metrics["MAE"])
            all_r2[name].append(metrics["R2"])

    def _mean_std(d: dict[str, list[float]]) -> tuple[dict[str, float], dict[str, float]]:
        mean_d = {name: float(np.mean(d[name])) for name in models}
        std_d = {
            name: float(np.std(d[name], ddof=1)) if len(d[name]) > 1 else 0.0
            for name in models
        }
        return mean_d, std_d

    mar_mean, mar_std = _mean_std(all_mar)
    mse_mean, mse_std = _mean_std(all_mse)
    rmse_mean, rmse_std = _mean_std(all_rmse)
    mae_mean, mae_std = _mean_std(all_mae)
    r2_mean, r2_std = _mean_std(all_r2)

    # Relative gain vs Linear baseline
    linear_mar = mar_mean.get("Linear")
    relative_gain_vs_linear: dict[str, float] = {}
    if linear_mar is not None and linear_mar > 0:
        for name in models:
            if name == "Linear":
                relative_gain_vs_linear[name] = 0.0
            else:
                from ensemble_see.metrics import relative_gain
                relative_gain_vs_linear[name] = relative_gain(linear_mar, mar_mean[name])

    # Wilcoxon: compare each model's MAR distribution vs Linear (across iterations)
    wilcoxon_vs_linear: dict[str, float] = {}
    if "Linear" in all_mar and len(all_mar["Linear"]) > 1:
        linear_errors = np.array(all_mar["Linear"])
        for name in models:
            if name == "Linear":
                wilcoxon_vs_linear[name] = 1.0
            else:
                try:
                    _, pval = wilcoxon_signed_rank(
                        linear_errors, np.array(all_mar[name]), alternative="greater"
                    )
                    wilcoxon_vs_linear[name] = pval
                except Exception:
                    wilcoxon_vs_linear[name] = float("nan")

    return ExperimentResult(
        dataset_name=dataset_config.name,
        n_iterations=n_iters,
        n_train=n_train,
        n_test=n_test,
        mar_mean=mar_mean,
        mar_std=mar_std,
        mse_mean=mse_mean,
        mse_std=mse_std,
        rmse_mean=rmse_mean,
        rmse_std=rmse_std,
        mae_mean=mae_mean,
        mae_std=mae_std,
        r2_mean=r2_mean,
        r2_std=r2_std,
        relative_gain_vs_linear=relative_gain_vs_linear,
        wilcoxon_vs_linear=wilcoxon_vs_linear,
    )


def format_results(result: ExperimentResult) -> str:
    """Human-readable summary of experiment result."""
    lines = [
        f"Dataset: {result.dataset_name}",
        f"Iterations: {result.n_iterations}  Train: {result.n_train}  Test: {result.n_test}",
        "",
    ]
    names = sorted(result.mar_mean.keys())

    for label, mean_d, std_d in [
        ("MAR (mean ± std)", result.mar_mean, result.mar_std),
        ("MSE (mean ± std)", result.mse_mean, result.mse_std),
        ("RMSE (mean ± std)", result.rmse_mean, result.rmse_std),
        ("MAE (mean ± std)", result.mae_mean, result.mae_std),
        ("R² (mean ± std)", result.r2_mean, result.r2_std),
    ]:
        if not mean_d:
            continue
        lines.append(f"{label}:")
        for name in names:
            m, s = mean_d.get(name, 0.0), std_d.get(name, 0.0)
            lines.append(f"  {name:12}  {m:.6f} ± {s:.6f}")
        lines.append("")

    if result.relative_gain_vs_linear:
        lines.append("Relative Gain vs Linear (%):")
        for name in sorted(result.relative_gain_vs_linear.keys()):
            if name != "Linear":
                lines.append(f"  {name:12}  {result.relative_gain_vs_linear[name]:+.2f}%")
        lines.append("")
    if result.wilcoxon_vs_linear:
        lines.append("Wilcoxon p-value (H1: model < Linear):")
        for name in sorted(result.wilcoxon_vs_linear.keys()):
            if name != "Linear":
                p = result.wilcoxon_vs_linear[name]
                lines.append(f"  {name:12}  p = {p:.4f}")
    return "\n".join(lines).rstrip()
