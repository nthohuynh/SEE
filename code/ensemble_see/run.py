"""
CLI entry point to run SEE ensemble experiments on configured datasets.

Usage:
    python -m ensemble_see.run [--datasets albrecht kemerer] [--iterations 1000] [--max-iter 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package is importable when run as script
if __name__ == "__main__" and __package__ is None:
    _code = Path(__file__).resolve().parents[1]
    if str(_code) not in sys.path:
        sys.path.insert(0, str(_code))
    __package__ = "ensemble_see"

from ensemble_see.config import DATASET_REGISTRY, ExperimentConfig
from ensemble_see.experiment import run_experiment, format_results


def main() -> int:
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
        help="Skip ensemble models (Bagging/Stacking)",
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
    args = parser.parse_args()

    if args.no_baselines and args.no_ensemble:
        print("Error: at least one of baselines or ensemble must be run", file=sys.stderr)
        return 1

    for key in args.datasets:
        if key not in DATASET_REGISTRY:
            print(f"Error: unknown dataset '{key}'. Available: {list(DATASET_REGISTRY)}", file=sys.stderr)
            return 1

    exp_config = ExperimentConfig(
        n_iterations=args.iterations,
        train_ratio=args.train_ratio,
        random_state=args.seed,
        max_iterations=args.max_iter,
        use_pso=args.use_pso,
        pso_n_particles=args.pso_particles,
        pso_n_iterations=args.pso_iterations,
        pso_cv_folds=args.pso_cv_folds,
    )

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
