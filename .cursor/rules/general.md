# Rules

This file provides guidance to agents when working in this repo.

## Project Overview

**OTP-FM** (Optimal Transport Potentials for Multi-Marginal Flow Matching) extends flow-matching to handle intermediate marginal constraints at specified times t_k ∈ (0, 1). The method incorporates optimal transport potential terms into the dynamic OT framework.

This repository provides:
1. An installable `otpfm` Python package for users
2. Reproducible experiments for the ICML paper

## Development Commands

```bash
# Install dependencies using pixi
pixi install
pixi shell

# Run experiments
python experiments/train.py --dataset gaussian --config configs/gaussian/defaults.json
```

## Testing

**Run tests after making changes:**

```bash
# Run all tests
pixi run pytest tests/ -v

# Run specific test files
pixi run pytest tests/test_gaussian.py -v
pixi run pytest tests/test_singlecell.py -v
pixi run pytest tests/test_gulfofmexico.py -v
pixi run pytest tests/test_beijingair.py -v
```

**Linting:**

```bash
pixi run ruff check src/otpfm/ experiments/
pixi run black --check src/otpfm/ experiments/
```

**When to run tests:**
- After modifying `src/otpfm/` (model, potentials, networks, solvers)
- After modifying `experiments/` (trainers, data loading, evaluation)

## Architecture

### Core Package: `otpfm` (`src/otpfm/`)

The installable package containing the core algorithm:

```python
from otpfm import OTPFM
from otpfm.potentials import W2InfPotential, W2Potential, EntropicW2Potential, KLPotential
from otpfm.lambdas import GaussianLambda, DeltaLambda
from otpfm.solvers import AndersonSolver
from otpfm.networks import FlowNetMLP
```

#### `OTPFM` (`src/otpfm/otpfm.py`)

The main model that learns velocity fields v(x, t₁, t₂) using MeanFlow consistency loss with OT potential corrections:

- **MeanFlow Loss**: Uses JVP to enforce consistency between velocity predictions at different time windows
- **OT Potential Corrections**: Solves self-consistent fixed-point equations for X_tk
- **Progressive Training**: `otp_alpha` gradually transitions from pure MeanFlow (0) to full OT corrections (1)
- **EMA Model**: Maintains exponential moving average of weights for stable sampling

#### Potentials (`src/otpfm/potentials.py`)

Abstract `Potential` class with implementations:
- `W2InfPotential`: Random coupling gradient (X_tk - x_true), fast and recommended
- `W2Potential`: Exact Wasserstein-2 OT map (requires `pot`, slow)
- `EntropicW2Potential`: GPU-accelerated Sinkhorn OT
- `KLPotential`: KL divergence with sliced or KDE estimation
- `MMDRBFPotential`, `MMDPolyPotential`: Maximum Mean Discrepancy variants

Each potential has a `LambdaFunction` defining time-localization: `GaussianLambda` (default), `DeltaLambda`, `TriangleLambda`, `BoxLambda`.

#### Fixed-Point Solvers (`src/otpfm/solvers.py`)

Solvers for the coupled X_tk equations:
- `DirectSolver`: Direct solve for linear case
- `PicardSolver`: Simple iteration (baseline)
- `AndersonSolver`: Anderson acceleration (recommended)
- `AndersonSafe`: Safeguarded with adaptive damping
- `AndersonHomotopy`: Continuation method for high-strength potentials

#### Network (`src/otpfm/networks.py`)

`FlowNetMLP`: MLP with positional time embeddings predicting velocity v(x, t, dt).

### Experiments (`experiments/`)

Reproducible experiments organized by dataset:

- `experiments/common/`: Shared utilities (Trainer, evaluation, plotting)
- `experiments/gaussian/`: Gaussian experiments with exact solver
- `experiments/singlecell/`: EB single-cell trajectory inference
- `experiments/gulfofmexico/`: Ocean current modeling
- `experiments/beijingair/`: PM2.5 air quality forecasting
- `experiments/baselines/`: Wrapper scripts for MMFM and 3MSBM
- `experiments/train.py`: Unified training entry point

**Running a single experiment:**

```bash
# Run with dataset defaults only
python experiments/train.py --dataset singlecell

# Run with specific potential config (layers on top of defaults)
python experiments/train.py --dataset singlecell --potential W2Inf

# Override specific options via CLI
python experiments/train.py --dataset gulfofmexico --potential KL --epochs 500 --seed 42

# Use a custom config file
python experiments/train.py --dataset singlecell --config path/to/custom.json

# List all available config options
python experiments/train.py --list-options
```

Available datasets: `gaussian`, `singlecell`, `gulfofmexico`, `beijingair`
Available potentials: `W2Inf` (default), `W2`, `KL`, `MMD`

**Running multiple seeds:**

```bash
# Usage: ./experiments/scripts/run_config.sh <dataset> <potential> [num_seeds] [date_tag]
./experiments/scripts/run_config.sh singlecell W2Inf 5 26Jan24

# Results saved to: results_local/<date_tag>/<potential>/seed_*/
```

**Post-training evaluation (re-run metrics on trained models):**

```bash
./experiments/scripts/run_config.sh --post-training singlecell W2Inf 26Jan24
```


### Notebooks (`notebooks/`)

Tutorial notebooks demonstrating the package:
- `01_quickstart_gaussians.ipynb`: Basic OTP-FM usage
- `02_singlecell_eb.ipynb`: Single-cell trajectory inference
- `03_gulf_of_mexico.ipynb`: Ocean current modeling
- `04_beijing_airquality.ipynb`: Air quality forecasting
- `05_exact_gaussian_solutions.ipynb`: Analytical Gaussian solutions

## Key Concepts

- **tks**: List of intermediate time points where marginal constraints are enforced
- **strength**: Potential strength controlling how strongly paths are pulled toward intermediate marginals
- **lambda_type**: Time localization shape ("gaussian" default, "delta", "triangle", "box")
- **width**: Half-width of lambda functions (default 0.2)
- **otp_alpha**: Progressive loss weight (0 = MeanFlow only, 1 = full OT corrections)

## Workflow

1. Define marginal distributions (source, intermediates, target)
2. Create `Potential` objects with desired type, strength, and time localization
3. Instantiate `OTPFM` model with potentials in an `OrderedDict[float, Potential]`
4. Use a `Trainer` to train with progressive loss weighting
5. Sample trajectories with `model.sample(x0s, n_steps)`

## Codebase Layout

```
src/otpfm/              # Installable package
├── __init__.py         # Exports OTPFM
├── otpfm.py            # Core model
├── potentials.py       # Potential types (W2InfPotential, etc.)
├── lambdas.py # Time localizations
├── solvers.py          # Fixed-point solvers
└── networks.py         # FlowNetMLP

experiments/            # Reproducible experiments (not installed)
├── common/             # Shared utilities
│   ├── trainer.py      # Base Trainer class
│   ├── evaluation.py   # Metrics (SWD, MMD, FGD, W2)
│   └── plotting.py     # Common plotting functions
├── gaussian/           # Gaussian experiments
├── singlecell/         # EB single-cell
├── gulfofmexico/       # Ocean currents
├── beijingair/         # PM2.5 air quality
├── baselines/          # MMFM, 3MSBM wrappers
├── train.py            # Unified training script
└── compare.py          # Comparison script

notebooks/              # Tutorial notebooks
configs/                # JSON configuration files
tests/                  # pytest tests
final_trajectories/     # Pre-computed results (not committed)
```

## Important Notes

- Run `pixi run ruff check` and `pixi run pytest` before committing
- Don't jump through hoops for backwards compatibility
