"""
Monte Carlo experiment loop for SEE ensemble study.

Follows plan/ensemble_regression_see.md (Carvalho et al., IJSEA 2020):
- Shuffle → Split 70% train / 30% test → Train on train only → Predict on test → MAR.
- Normalization: Max-Min [0,1] fit on training data only, then transform train and test
  (no leakage). StackingRegressor uses cv=5 for meta-learner (out-of-fold base predictions).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any
import warnings

import numpy as np
import pandas as pd
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
from ensemble_see.gwo_optimizer import GWOConfig, GWOHyperparameterOptimizer


def _debug_log(
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    # region agent log
    payload = {
        "sessionId": "beec08",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        log_path = Path(__file__).resolve().parents[2] / "debug-beec08.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass
    # endregion


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
    mmre_mean: dict[str, float] = field(default_factory=dict)
    mmre_std: dict[str, float] = field(default_factory=dict)
    pred25_mean: dict[str, float] = field(default_factory=dict)
    pred25_std: dict[str, float] = field(default_factory=dict)
    relative_gain_vs_linear: dict[str, float] = field(default_factory=dict)
    wilcoxon_vs_linear: dict[str, float] = field(default_factory=dict)
    """p-value of Wilcoxon test (errors_linear vs errors_model); 'less' alternative."""
    pso_hyperparameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    """PSO-optimized hyperparameters per model (model_name -> {param: value}). Only set when --use-pso."""
    gwo_hyperparameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    """GWO-optimized hyperparameters per model (model_name -> {param: value}). Only set when --use-gwo."""


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
        X_train_input: Any = X_train_s
        X_test_input: Any = X_test_s
        if name == "LGBM":
            # Keep feature names consistent for sklearn/lightgbm wrapper checks.
            feature_cols = [f"f{i}" for i in range(np.asarray(X_train_s).shape[1])]
            X_train_input = pd.DataFrame(X_train_s, columns=feature_cols)
            X_test_input = pd.DataFrame(X_test_s, columns=feature_cols)
        if name == "LGBM":
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H1",
                location="experiment.py:_run_single_iteration:before_fit",
                message="LGBM input before fit",
                data={
                    "x_train_type": type(X_train_input).__name__,
                    "y_train_type": type(y_train_s).__name__,
                    "x_test_type": type(X_test_input).__name__,
                    "x_train_shape": list(np.asarray(X_train_input).shape),
                    "x_test_shape": list(np.asarray(X_test_input).shape),
                },
            )
            # endregion
        model.fit(X_train_input, y_train_s)
        if name == "LGBM":
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H2",
                location="experiment.py:_run_single_iteration:after_fit",
                message="LGBM model metadata after fit",
                data={
                    "has_feature_name_in": hasattr(model, "feature_name_"),
                    "feature_name_len": len(getattr(model, "feature_name_", []))
                    if hasattr(model, "feature_name_")
                    else 0,
                },
            )
            # endregion
        if name == "LGBM":
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                y_pred = model.predict(X_test_input)
            # region agent log
            _debug_log(
                run_id="post-fix",
                hypothesis_id="H8",
                location="experiment.py:_run_single_iteration:predict_warning_check",
                message="Warnings captured around LGBM predict",
                data={
                    "warning_count": len(caught_warnings),
                    "warning_messages": [str(w.message) for w in caught_warnings[:3]],
                },
            )
            # endregion
        else:
            y_pred = model.predict(X_test_input)
        if name == "LGBM":
            # region agent log
            _debug_log(
                run_id="pre-fix",
                hypothesis_id="H3",
                location="experiment.py:_run_single_iteration:after_predict",
                message="LGBM predict completed",
                data={
                    "y_pred_type": type(y_pred).__name__,
                    "y_pred_shape": list(np.asarray(y_pred).shape),
                },
            )
            # endregion
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
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H4",
        location="experiment.py:run_experiment:dataset_loaded",
        message="Dataset loaded for experiment",
        data={
            "dataset": dataset_config.name,
            "x_type": type(X).__name__,
            "y_type": type(y).__name__,
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]) if len(X.shape) > 1 else 0,
            "feature_preview": frame.feature_names[:5],
        },
    )
    # endregion
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
            convergence_plot_dir=experiment_config.pso_convergence_plot_dir,
        )
        optimizer = PSOHyperparameterOptimizer(pso_config)

        # Get builder functions for ensemble models
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
        builder_map = {
            "B-LR": build_bagging_lr,
            "B-RR": build_bagging_rr,
            "B-RI": build_bagging_ri,
            "B-LA": build_bagging_la,
            "ST-LR": build_stacking_lr,
            "ST-RR": build_stacking_rr,
            "ST-RI": build_stacking_ri,
            "ST-LA": build_stacking_la,
            "GBR": build_gradient_boosting,
            "HGB": build_hist_gradient_boosting,
            "XGB": build_xgboost,
            "LGBM": build_lightgbm,
            "CAT": build_catboost,
        }
        ensemble_models_builders = {
            name: builder_map[name]
            for name in models.keys()
            if name in builder_map
        }

        if ensemble_models_builders:
            pso_results = optimizer.optimize_multiple_models(
                ensemble_models_builders, X_train_pso_s, y_train_pso_s,
                dataset_name=dataset_config.name,
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

    # GWO hyperparameter optimization (if enabled)
    optimized_gwo_hyperparams: dict[str, dict[str, Any]] = {}
    if experiment_config.use_gwo and include_ensemble:
        # Use a single train/test split for GWO optimization
        rng_gwo = np.random.default_rng(experiment_config.random_state)
        perm_gwo = rng_gwo.permutation(n)
        X_shuf_gwo, y_shuf_gwo = X[perm_gwo], y[perm_gwo]
        X_train_gwo, X_test_gwo, y_train_gwo, y_test_gwo = train_test_split(
            X_shuf_gwo, y_shuf_gwo,
            train_ratio=experiment_config.train_ratio,
            random_state=experiment_config.random_state,
        )
        X_train_gwo_s, y_train_gwo_s, _, _ = scale_fit_train_transform_test(
            X_train_gwo, y_train_gwo, X_test_gwo, y_test_gwo
        )

        gwo_config = GWOConfig(
            n_wolves=experiment_config.gwo_n_wolves,
            n_iterations=experiment_config.gwo_n_iterations,
            cv_folds=experiment_config.gwo_cv_folds,
            random_state=experiment_config.random_state,
            verbose=True,
            convergence_plot_dir=experiment_config.gwo_convergence_plot_dir,
        )
        gwo_optimizer = GWOHyperparameterOptimizer(gwo_config)

        # Get builder functions for ensemble models
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
        gwo_builder_map = {
            "B-LR":  build_bagging_lr,
            "B-RR":  build_bagging_rr,
            "B-RI":  build_bagging_ri,
            "B-LA":  build_bagging_la,
            "ST-LR": build_stacking_lr,
            "ST-RR": build_stacking_rr,
            "ST-RI": build_stacking_ri,
            "ST-LA": build_stacking_la,
            "GBR":   build_gradient_boosting,
            "HGB":   build_hist_gradient_boosting,
            "XGB":   build_xgboost,
            "LGBM":  build_lightgbm,
            "CAT":   build_catboost,
        }
        gwo_ensemble_builders = {
            name: gwo_builder_map[name]
            for name in models.keys()
            if name in gwo_builder_map
        }

        if gwo_ensemble_builders:
            gwo_results = gwo_optimizer.optimize_multiple_models(
                gwo_ensemble_builders, X_train_gwo_s, y_train_gwo_s,
                dataset_name=dataset_config.name,
            )
            optimized_gwo_hyperparams = {
                name: params for name, (params, _) in gwo_results.items()
            }
            # Update models with GWO-optimized hyperparameters
            from ensemble_see.gwo_optimizer import _get_hyperparameter_configs as _get_gwo_configs
            gwo_configs = _get_gwo_configs()
            for name, hyperparams in optimized_gwo_hyperparams.items():
                if name in models and hyperparams and name in gwo_configs:
                    base_model = models[name]
                    optimized_model = clone(base_model)
                    models[name] = gwo_configs[name].param_setter(optimized_model, hyperparams)

    n_iters = experiment_config.effective_iterations()
    rng = np.random.default_rng(experiment_config.random_state)
    iteration_seeds = rng.integers(0, 2**31, size=n_iters)

    all_mar: dict[str, list[float]] = {name: [] for name in models}
    all_mse: dict[str, list[float]] = {name: [] for name in models}
    all_rmse: dict[str, list[float]] = {name: [] for name in models}
    all_mae: dict[str, list[float]] = {name: [] for name in models}
    all_r2: dict[str, list[float]] = {name: [] for name in models}
    all_mmre: dict[str, list[float]] = {name: [] for name in models}
    all_pred25: dict[str, list[float]] = {name: [] for name in models}
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
            all_mmre[name].append(metrics["MMRE"])
            all_pred25[name].append(metrics["Pred25"])

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
    mmre_mean, mmre_std = _mean_std(all_mmre)
    pred25_mean, pred25_std = _mean_std(all_pred25)

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
        mmre_mean=mmre_mean,
        mmre_std=mmre_std,
        pred25_mean=pred25_mean,
        pred25_std=pred25_std,
        relative_gain_vs_linear=relative_gain_vs_linear,
        wilcoxon_vs_linear=wilcoxon_vs_linear,
        pso_hyperparameters=optimized_hyperparams,
        gwo_hyperparameters=optimized_gwo_hyperparams,
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
        ("MMRE (mean ± std)", result.mmre_mean, result.mmre_std),
        ("Pred25 (mean ± std)", result.pred25_mean, result.pred25_std),
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


def result_to_csv_rows(result: ExperimentResult) -> list[dict[str, Any]]:
    """Convert experiment result to list of row dicts (one per model) for CSV export."""
    names = sorted(result.mar_mean.keys())
    rows = []
    for name in names:
        row = {
            "dataset": result.dataset_name,
            "model": name,
            "n_iterations": result.n_iterations,
            "n_train": result.n_train,
            "n_test": result.n_test,
            "MAR_mean": result.mar_mean.get(name),
            "MAR_std": result.mar_std.get(name),
            "MSE_mean": result.mse_mean.get(name),
            "MSE_std": result.mse_std.get(name),
            "RMSE_mean": result.rmse_mean.get(name),
            "RMSE_std": result.rmse_std.get(name),
            "MAE_mean": result.mae_mean.get(name),
            "MAE_std": result.mae_std.get(name),
            "R2_mean": result.r2_mean.get(name),
            "R2_std": result.r2_std.get(name),
            "MMRE_mean": result.mmre_mean.get(name),
            "MMRE_std": result.mmre_std.get(name),
            "Pred25_mean": result.pred25_mean.get(name),
            "Pred25_std": result.pred25_std.get(name),
        }
        if result.relative_gain_vs_linear:
            row["relative_gain_vs_linear_pct"] = result.relative_gain_vs_linear.get(name)
        if result.wilcoxon_vs_linear:
            row["wilcoxon_p_vs_linear"] = result.wilcoxon_vs_linear.get(name)
        rows.append(row)
    return rows


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return list(rows[0].keys())


def save_result_to_csv(result: ExperimentResult, csv_path: Path, *, write_header: bool = True) -> None:
    """Append experiment result rows to a CSV file."""
    rows = result_to_csv_rows(result)
    if not rows:
        return
    fieldnames = _csv_fieldnames(rows)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_pso_hyperparameters(
    result: ExperimentResult, csv_path: Path, *, write_header: bool = True
) -> None:
    """Append PSO-optimized hyperparameters (dataset, model, hyperparameters JSON) to a CSV file."""
    if not result.pso_hyperparameters:
        return
    rows = []
    for model_name in sorted(result.pso_hyperparameters.keys()):
        params = result.pso_hyperparameters[model_name]
        if not params:
            continue
        rows.append({
            "dataset": result.dataset_name,
            "model": model_name,
            "hyperparameters": json.dumps(params, sort_keys=True),
        })
    if not rows:
        return
    fieldnames = ["dataset", "model", "hyperparameters"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_gwo_hyperparameters(
    result: ExperimentResult, csv_path: Path, *, write_header: bool = True
) -> None:
    """Append GWO-optimized hyperparameters (dataset, model, hyperparameters JSON) to a CSV file."""
    if not result.gwo_hyperparameters:
        return
    rows = []
    for model_name in sorted(result.gwo_hyperparameters.keys()):
        params = result.gwo_hyperparameters[model_name]
        if not params:
            continue
        rows.append({
            "dataset": result.dataset_name,
            "model": model_name,
            "hyperparameters": json.dumps(params, sort_keys=True),
        })
    if not rows:
        return
    fieldnames = ["dataset", "model", "hyperparameters"]
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_figures(result: ExperimentResult, figures_dir: Path) -> None:
    """Save bar charts of metrics (mean ± std) per model to figures_dir."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(result.mar_mean.keys())
    if not names:
        return

    metrics_config = [
        ("MAR", result.mar_mean, result.mar_std),
        ("MMRE", result.mmre_mean, result.mmre_std),
        ("Pred25", result.pred25_mean, result.pred25_std),
        ("MSE", result.mse_mean, result.mse_std),
        ("RMSE", result.rmse_mean, result.rmse_std),
        ("MAE", result.mae_mean, result.mae_std),
        ("R2", result.r2_mean, result.r2_std),
    ]
    for metric_label, mean_d, std_d in metrics_config:
        if not mean_d:
            continue
        means = [mean_d.get(n, 0) for n in names]
        stds = [std_d.get(n, 0) for n in names]
        # Replace nan for plotting
        means = [m if m == m else 0.0 for m in means]  # noqa: E721
        stds = [s if s == s else 0.0 for s in stds]  # noqa: E721

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5), 5))
        x = range(len(names))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color="steelblue", edgecolor="navy", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{result.dataset_name} — {metric_label} (mean ± std)")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out = figures_dir / f"{result.dataset_name}_{metric_label}.png"
        plt.savefig(out, dpi=150)
        plt.close()