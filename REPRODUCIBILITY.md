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

- **Single-cell (Embryoid Body)**: Downloads from the [TrajectoryNet repository](https://github.com/KrishnaswamyLab/TrajectoryNet/raw/master/data/eb_velocity_v5.npz)
- **Gulf of Mexico**: Downloads from the [SB-IRR repository](https://github.com/YunyiShen/SB-Iterative-Reference-Refinement)
- **Beijing Air Quality**: Downloads from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
- **Gaussians**: Generated synthetically

## Reproducing paper results

```bash
python experiments/train.py --dataset {singlecell, beijingair, gulfofmexico} --potential {W2Inf, W2, MMD, KL}
```

This will automatically load the configurations used in the paper. For further tunable options, run:

```bash
python experiments/train.py -h
```

## Tutorial Notebooks

Interactive tutorials demonstrating each experiment:

1. `notebooks/01_quickstart_gaussians.ipynb` - Introduction with Gaussian data
2. `notebooks/02_singlecell_eb.ipynb` - Embryoid body single-cell trajectory inference
3. `notebooks/03_gulf_of_mexico.ipynb` - Ocean current modeling
4. `notebooks/04_beijing_airquality.ipynb` - Air quality forecasting
5. `notebooks/05_exact_gaussian_solutions.ipynb` - Analytical solutions
