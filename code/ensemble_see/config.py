"""
Configuration for SEE ensemble experiments.

Centralizes dataset paths, target/feature columns, and experiment parameters
so they can be changed in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Default root for datasets (relative to this file)
_DATASETS_ROOT = Path(__file__).resolve().parents[2] / "datasets"
_CLEANED = _DATASETS_ROOT / "cleanedData"


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for a single SEE dataset."""

    name: str
    path: Path
    target_column: str
    drop_columns: Sequence[str] = ()
    """Column names to exclude from features (e.g. ID, Project)."""

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            object.__setattr__(self, "path", Path(self.path))


# Predefined dataset configs for cleaned CSVs
DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "albrecht": DatasetConfig(
        name="albrecht",
        path=_CLEANED / "albrecht_processed.csv",
        target_column="Effort",
        drop_columns=(),
    ),
    "kemerer": DatasetConfig(
        name="kemerer",
        path=_CLEANED / "kemerer_processed.csv",
        target_column="EffortMM",
        drop_columns=("ID",),
    ),
}


@dataclass
class ExperimentConfig:
    """Parameters for the Monte Carlo ensemble experiment."""

    n_iterations: int = 1000
    train_ratio: float = 0.70
    random_state: int | None = 42
    n_jobs: int = -1
    """Parallel jobs for fitting (e.g. Bagging). -1 = all cores."""

    # Optional overrides for quick tests
    max_iterations: int | None = None
    """If set, cap n_iterations (e.g. 5 for smoke test)."""

    def effective_iterations(self) -> int:
        if self.max_iterations is not None:
            return min(self.n_iterations, self.max_iterations)
        return self.n_iterations
