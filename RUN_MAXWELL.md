# Running Maxwell Dataset with UV

This guide shows you how to run the maxwell dataset experiment using `uv` (fast Python package manager).

## Prerequisites

- `uv` installed ([install guide](https://github.com/astral-sh/uv#installation))
- Python >= 3.10

## Step 1: Install Dependencies

First, sync dependencies using `uv`:

```bash
cd /Users/duy/VKU/SoftwareEngineering/SEE/SEE
uv sync
```

This will:
- Create a virtual environment (if needed)
- Install all dependencies from `pyproject.toml`
- Lock versions in `uv.lock`

## Step 2: Basic Run (Maxwell Dataset)

Run the maxwell dataset with default settings:

```bash
uv run python -m ensemble_see.run --datasets maxwell
```

This will:
- Run 1000 Monte Carlo iterations
- Use 70% train / 30% test split
- Include both baseline and ensemble models
- Use random seed 42

## Step 3: Quick Test Run

For a quick test (e.g., 5 iterations):

```bash
uv run python -m ensemble_see.run --datasets maxwell --max-iter 5
```

## Step 4: Run with PSO Optimization

Enable PSO hyperparameter optimization:

```bash
uv run python -m ensemble_see.run --datasets maxwell --use-pso
```

With custom PSO parameters:

```bash
uv run python -m ensemble_see.run \
  --datasets maxwell \
  --use-pso \
  --pso-particles 30 \
  --pso-iterations 50 \
  --pso-cv-folds 5
```

## Step 5: Run Only Ensemble Models (No Baselines)

```bash
uv run python -m ensemble_see.run --datasets maxwell --no-baselines
```

## Step 6: Run Only Baselines (No Ensemble)

```bash
uv run python -m ensemble_see.run --datasets maxwell --no-ensemble
```

## Step 7: Custom Configuration

Full example with custom settings:

```bash
uv run python -m ensemble_see.run \
  --datasets maxwell \
  --iterations 500 \
  --train-ratio 0.75 \
  --seed 123 \
  --use-pso \
  --pso-particles 20 \
  --pso-iterations 30 \
  --pso-cv-folds 3
```

## Step 8: Run Multiple Datasets

Run maxwell along with other datasets:

```bash
uv run python -m ensemble_see.run --datasets maxwell albrecht kemerer
```

## Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--datasets` | Dataset names to run | `albrecht kemerer` |
| `--iterations` | Monte Carlo iterations | `1000` |
| `--max-iter` | Cap iterations for quick test | `None` |
| `--train-ratio` | Train split ratio | `0.70` |
| `--seed` | Random seed | `42` |
| `--no-baselines` | Skip baseline models | `False` |
| `--no-ensemble` | Skip ensemble models | `False` |
| `--use-pso` | Enable PSO optimization | `False` |
| `--pso-particles` | PSO swarm size | `30` |
| `--pso-iterations` | PSO iterations | `50` |
| `--pso-cv-folds` | CV folds for PSO | `5` |

## Troubleshooting

### If dependencies are missing:

```bash
uv sync --upgrade
```

### If numpy build fails (BLAS library missing):

On macOS, install OpenBLAS:
```bash
brew install openblas
```

Or use pre-built wheels by updating numpy version:
```bash
uv add "numpy>=1.24,<3" --no-build-isolation
```

### If you need to add a new dependency:

```bash
uv add package-name
```

### Check installed packages:

```bash
uv pip list
```

### Run Python interactively with the environment:

```bash
uv run python
```

Then import:
```python
from ensemble_see.config import DATASET_REGISTRY
print(list(DATASET_REGISTRY.keys()))  # ['albrecht', 'kemerer', 'china', 'maxwell']
```

### Verify maxwell dataset exists:

```bash
ls -la datasets/maxwell.arff
```

## Example Output

When you run the experiment, you'll see output like:

```
============================================================
Running: maxwell
============================================================
Dataset: maxwell
Iterations: 1000  Train: XX  Test: XX

MAR (mean ± std):
  B-LA         0.XXXXXX ± 0.XXXXXX
  B-LR         0.XXXXXX ± 0.XXXXXX
  B-RI         0.XXXXXX ± 0.XXXXXX
  B-RR         0.XXXXXX ± 0.XXXXXX
  ELM-2        0.XXXXXX ± 0.XXXXXX
  ELM-5        0.XXXXXX ± 0.XXXXXX
  Linear       0.XXXXXX ± 0.XXXXXX
  ST-LA        0.XXXXXX ± 0.XXXXXX
  ST-LR        0.XXXXXX ± 0.XXXXXX
  ST-RI        0.XXXXXX ± 0.XXXXXX
  ST-RR        0.XXXXXX ± 0.XXXXXX
...
```

## Notes

- The maxwell dataset is loaded from `datasets/maxwell.arff`
- Results show MAR (Mean Absolute Residual) as the primary metric
- PSO optimization runs before the Monte Carlo experiment
- Each iteration uses a different random shuffle for train/test split
