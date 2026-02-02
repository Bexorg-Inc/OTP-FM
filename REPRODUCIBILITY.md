# Reproducing Paper Experiments

This guide explains how to reproduce all experiments from the ICML 2025 paper "Optimal Transport Potentials for Multi-Marginal Flow Matching".

## Setup

Install the package with experiment dependencies:

```bash
# Using pip
pip install otpfm[experiments]

# Using pixi (recommended for development)
pixi install
pixi shell
```

## Datasets

All datasets are automatically downloaded during training. Storage locations:

- **Single-cell (Embryoid Body)**: Downloads from TrajectoryNet repository
- **Gulf of Mexico**: Downloads from GitHub
- **Beijing Air Quality**: Downloads from UCI repository  
- **Gaussians**: Generated synthetically

## Running Experiments

### 1. Gaussian Experiments

Quick demonstration of OTP-FM on synthetic Gaussian data:

```bash
# Train OTP-FM on 1D Gaussians
python experiments/train.py --config experiments/gaussian/configs/otpfm_1d.json

# Train baseline (flow matching without potentials)
python experiments/train.py --config experiments/gaussian/configs/baseline_1d.json
```

### 2. Single-cell Trajectory Inference

Train on Embryoid Body (EB) differentiation data:

```bash
# OTP-FM with W2 potential
python experiments/train.py --config experiments/singlecell/configs/otpfm_w2.json

# OTP-FM with KL potential
python experiments/train.py --config experiments/singlecell/configs/otpfm_kl.json

# OTP-FM with MMD potential
python experiments/train.py --config experiments/singlecell/configs/otpfm_mmd.json
```

### 3. Gulf of Mexico (Ocean Currents)

Train on ocean drifter data:

```bash
# OTP-FM
python experiments/train.py --config experiments/gulfofmexico/configs/otpfm.json

# Baseline
python experiments/train.py --config experiments/gulfofmexico/configs/baseline.json
```

### 4. Beijing Air Quality

Train on PM2.5 forecasting data:

```bash
# OTP-FM
python experiments/train.py --config experiments/beijingair/configs/otpfm.json

# Baseline  
python experiments/train.py --config experiments/beijingair/configs/baseline.json
```

## Evaluation and Comparison

After training, evaluate models and generate comparison plots:

```bash
# Evaluate single-cell experiments
python experiments/compare.py --dataset singlecell --trajectories-dir final_trajectories/

# Evaluate all datasets
python experiments/compare.py --all
```

This generates:
- Metric tables (SWD, MMD, FGD, W2) in CSV and LaTeX format
- PCA trajectory comparison plots
- Multi-seed aggregation with mean ± std

## Baseline Methods

To run baseline comparisons (MMFM, 3MSBM):

```bash
# Setup baseline repositories
cd experiments/baselines
./setup.sh

# Run MMFM baseline
python run_mmfm.py --dataset singlecell

# Run 3MSBM baseline  
python run_3msbm.py --dataset singlecell
```

See `experiments/baselines/README.md` for detailed setup instructions.

## Tutorial Notebooks

Interactive tutorials demonstrating each experiment:

1. `notebooks/01_quickstart_gaussians.ipynb` - Introduction with Gaussian data
2. `notebooks/02_singlecell_eb.ipynb` - Single-cell trajectory inference
3. `notebooks/03_gulf_of_mexico.ipynb` - Ocean current modeling
4. `notebooks/04_beijing_airquality.ipynb` - Air quality forecasting
5. `notebooks/05_exact_gaussian_solutions.ipynb` - Analytical solutions

## Hardware Requirements

- **Minimum**: 8GB GPU memory for single-cell/GoM/Beijing experiments
- **Recommended**: 16GB+ GPU for larger batch sizes
- **Gaussians**: Can run on CPU

Training times (single NVIDIA A100):
- Gaussians: ~5 minutes
- Single-cell: ~30 minutes
- Gulf of Mexico: ~20 minutes
- Beijing Air Quality: ~15 minutes

## Expected Results

| Dataset | Metric | OTP-FM (W2) | OTP-FM (KL) | Baseline |
|---------|--------|-------------|-------------|----------|
| Single-cell | SWD ↓ | 0.XXX | 0.XXX | 0.XXX |
| GoM | SWD ↓ | 0.XXX | 0.XXX | 0.XXX |
| Beijing | SWD ↓ | 0.XXX | 0.XXX | 0.XXX |

*Note: Exact values will vary slightly due to random initialization.*

## Troubleshooting

**Out of memory**: Reduce batch size in config file.

**Slow training**: Enable mixed precision by adding `"mixed_precision": true` to config.

**Missing data**: Check internet connection; datasets download automatically.
