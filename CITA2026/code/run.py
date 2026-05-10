"""
CLI entry point to run SEE ensemble experiments on configured datasets.

Usage:
    python -m ensemble_see.run [--datasets albrecht kemerer] [--iterations 1000] [--max-iter 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure package is importable when run as script
if __name__ == "__main__" and __package__ is None:
    _code = Path(__file__).resolve().parents[1]
    if str(_code) not in sys.path:
        sys.path.insert(0, str(_code))
    __package__ = "ensemble_see"

from ensemble_see.config import DATASET_REGISTRY, ExperimentConfig
from ensemble_see.experiment import (
    format_results,
    run_experiment,
    save_figures,
    save_gwo_hyperparameters,
    save_pso_hyperparameters,
    save_result_to_csv,
)


def _debug_log_run(
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
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


def main() -> int:
    # region agent log
    _debug_log_run(
        run_id="post-fix",
        hypothesis_id="H5",
        location="run.py:main:start",
        message="Entrypoint runtime context",
        data={
            "cwd": os.getcwd(),
            "run_file": str(Path(__file__).resolve()),
            "argv0": sys.argv[0] if sys.argv else "",
            "sys_path0": sys.path[0] if sys.path else "",
        },
    )
    # endregion
    parser = argparse.ArgumentParser(
        description="Run ensemble SEE experiment (Carvalho et al. reproduction)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["albrecht", "kemerer"],
        help="Dataset keys from config (default: albrecht kemerer)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Monte Carlo iterations (default: 1000)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Cap iterations for quick test (e.g. 5)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Train split ratio (default: 0.70)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline models (Linear, ELM-2, ELM-5)",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Skip ensemble models (Bagging/Stacking/Boosting)",
    )
    parser.add_argument(
        "--use-pso",
        action="store_true",
        help="Use PSO for hyperparameter optimization",
    )
    parser.add_argument(
        "--pso-particles",
        type=int,
        default=30,
        help="Number of PSO particles (default: 30)",
    )
    parser.add_argument(
        "--pso-iterations",
        type=int,
        default=50,
        help="Number of PSO iterations (default: 50)",
    )
    parser.add_argument(
        "--pso-cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for PSO (default: 5)",
    )
    parser.add_argument(
        "--use-gwo",
        action="store_true",
        help="Use GWO for hyperparameter optimization",
    )
    parser.add_argument(
        "--gwo-wolves",
        type=int,
        default=30,
        help="Number of GWO wolves in the pack (default: 30)",
    )
    parser.add_argument(
        "--gwo-iterations",
        type=int,
        default=50,
        help="Number of GWO hunting iterations (default: 50)",
    )
    parser.add_argument(
        "--gwo-cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for GWO (default: 5)",
    )
    args = parser.parse_args()

    if args.no_baselines and args.no_ensemble:
        print("Error: at least one of baselines or ensemble must be run", file=sys.stderr)
        return 1

    for key in args.datasets:
        if key not in DATASET_REGISTRY:
            print(f"Error: unknown dataset '{key}'. Available: {list(DATASET_REGISTRY)}", file=sys.stderr)
            return 1

    # Output paths (project root = parent of code/)
    project_root = Path(__file__).resolve().parents[2]
    figures_dir = project_root / "figures"
    csv_path = project_root / "evaluation_results.csv"
    pso_csv_path = project_root / "pso_hyperparameters.csv"
    figures_dir.mkdir(parents=True, exist_ok=True)
    write_csv_header = not csv_path.exists() or csv_path.stat().st_size == 0
    write_pso_header = not pso_csv_path.exists() or pso_csv_path.stat().st_size == 0

    gwo_csv_path = project_root / "gwo_hyperparameters.csv"
    write_gwo_header = not gwo_csv_path.exists() or gwo_csv_path.stat().st_size == 0

    exp_config = ExperimentConfig(
        n_iterations=args.iterations,
        train_ratio=args.train_ratio,
        random_state=args.seed,
        max_iterations=args.max_iter,
        use_pso=args.use_pso,
        pso_n_particles=args.pso_particles,
        pso_n_iterations=args.pso_iterations,
        pso_cv_folds=args.pso_cv_folds,
        pso_convergence_plot_dir=figures_dir,
        use_gwo=args.use_gwo,
        gwo_n_wolves=args.gwo_wolves,
        gwo_n_iterations=args.gwo_iterations,
        gwo_cv_folds=args.gwo_cv_folds,
        gwo_convergence_plot_dir=figures_dir,
    )

    # Print active configuration summary
    print("\n" + "="*60)
    print("EXPERIMENT CONFIGURATION")
    print("="*60)
    print(f"  Datasets        : {', '.join(args.datasets)}")
    print(f"  Iterations      : {exp_config.effective_iterations()} "
          f"(of {args.iterations}{ ', capped at ' + str(args.max_iter) if args.max_iter else ''})")
    print(f"  Train ratio     : {args.train_ratio:.0%} train / {1-args.train_ratio:.0%} test")
    print(f"  Random seed     : {args.seed}")
    print(f"  Baselines       : {'off' if args.no_baselines else 'on'}")
    print(f"  Ensemble        : {'off' if args.no_ensemble else 'on'}")
    print(f"  PSO optimizer   : {'ON  -- particles={}, iterations={}, cv_folds={}'.format(args.pso_particles, args.pso_iterations, args.pso_cv_folds) if args.use_pso else 'off'}")
    print(f"  GWO optimizer   : {'ON  -- wolves={}, iterations={}, cv_folds={}'.format(args.gwo_wolves, args.gwo_iterations, args.gwo_cv_folds) if args.use_gwo else 'off'}")
    print(f"  Output CSV      : {csv_path}")
    if args.use_pso:
        print(f"  PSO params CSV  : {pso_csv_path}")
    if args.use_gwo:
        print(f"  GWO params CSV  : {gwo_csv_path}")
    print(f"  Figures dir     : {figures_dir}")
    print("="*60)

    for key in args.datasets:
        ds_config = DATASET_REGISTRY[key]
        if not ds_config.path.exists():
            print(f"Warning: {ds_config.path} not found, skipping {key}", file=sys.stderr)
            continue
        print(f"\n{'='*60}\nRunning: {key}\n{'='*60}")
        result = run_experiment(
            ds_config,
            exp_config,
            include_baselines=not args.no_baselines,
            include_ensemble=not args.no_ensemble,
        )
        print(format_results(result))
        save_figures(result, figures_dir)
        save_result_to_csv(result, csv_path, write_header=write_csv_header)
        write_csv_header = False
        save_pso_hyperparameters(result, pso_csv_path, write_header=write_pso_header)
        if result.pso_hyperparameters:
            write_pso_header = False
            print(f"  PSO hyperparameters appended to {pso_csv_path}")
        save_gwo_hyperparameters(result, gwo_csv_path, write_header=write_gwo_header)
        if result.gwo_hyperparameters:
            write_gwo_header = False
            print(f"  GWO hyperparameters appended to {gwo_csv_path}")
        print(f"  Results appended to {csv_path}")
        print(f"  Figures saved to {figures_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())