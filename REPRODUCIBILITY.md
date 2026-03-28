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

### EB Leave-One-Out (WLF Paper Comparison)

This reproduces the Embryoid Body experiment from [Neklyudov et al. 2024 "A Computational Framework for Solving Wasserstein Lagrangian Flows"](https://arxiv.org/abs/2310.10649), Table 1. The experiment uses 5-dim PCA, leave-one-out cross-validation (holding out intermediate times t1, t2, or t3), and Wasserstein-1 distance in normalized space as the metric.

**Run all methods (W2 + W2Inf):**

```bash
./experiments/scripts/run_eb_loo.sh              # both W2 and W2Inf
./experiments/scripts/run_eb_loo.sh w2            # W2 only (OT-coupled, 768d/8L)
./experiments/scripts/run_eb_loo.sh w2inf         # W2Inf only (no OT coupling, 256d/4L)
./experiments/scripts/run_eb_loo.sh all --parallel # all methods, folds in parallel
```

Or run individual folds:

```bash
# W2 (OT-coupled, 768d/8L architecture)
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_fold1.json
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_fold2.json
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_fold3.json

# W2Inf (no OT coupling, 256d/4L architecture)
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_w2inf_fold1.json
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_w2inf_fold2.json
python experiments/train.py --dataset singlecell --config configs/singlecell/eb_loo_w2inf_fold3.json
```

**Expected results** (W1 on held-out marginal, normalized space):

| Method | Arch | Fold 1 (t1) | Fold 2 (t2) | Fold 3 (t3) | Average |
|--------|------|:-----------:|:-----------:|:-----------:|:-------:|
| **OTP-FM W2** | 768d/8L | **0.618** | **0.657** | **0.654** | **0.643** |
| **OTP-FM W2Inf** | 256d/4L | 0.712 | 0.732 | 0.801 | 0.748 |
| OT-CFM (WLF paper) | — | — | — | — | 0.822 |
| WLF-OT (WLF paper) | — | — | — | — | 0.641 |

**Notes:**
- W2 uses OT-coupled training (pre-computed alignments) with the full 768d/8L architecture (4.9M params). Fold 1 overfits after ~30 epochs; best checkpoint should be used.
- W2Inf uses random coupling with a smaller 256d/4L architecture (333K params), lower learning rate (5e-4), and 300 epochs. The smaller network reduces overfitting.
- `tks` (potential time points) are auto-computed as evenly spaced based on the number of intermediate training marginals.
- Config files are fully self-contained — no CLI overrides needed.

### EB 5D Leave-Two-Out (iJKOnet Paper Comparison)

This experiment uses **5-dim PCA**, holds out t1 and t3, trains on t0, t2, t4, and evaluates **W2 distance** (in normalized space) at the held-out times.

**Config:** `configs/singlecell/eb_ijko.json` (layers on top of `defaults.json` → `W2.json`)

**Run command:**

```bash
python experiments/train.py --dataset singlecell --potential w2 \
    --config configs/singlecell/eb_ijko.json --tag ijko_w2
```

**Notes:**
- W2 is computed in normalized (standardized) space, consistent with the iJKOnet paper and previous benchmarks.

### EB 100D Leave-One-Out (DMSB Paper Comparison)

This reproduces the Embryoid Body experiment from [Chen et al. 2023 "Deep Multi-Marginal Momentum Schrödinger Bridge"](https://arxiv.org/abs/2303.01751), Table 3. The experiment uses **100-dim PCA** with the standard single-cell config (`defaults.json`), evaluating MMD, SWD, and FGD at each timepoint. Four conditions: train-on-all, and leave-out t1/t2/t3.

**Configs:** The experiment-specific configs layer on top of `defaults.json` → `W2.json`:

- `configs/singlecell/eb_dmsb_all.json` — train on all 5 timepoints (no holdout)
- `configs/singlecell/eb_dmsb_lo_t1.json` — hold out t1
- `configs/singlecell/eb_dmsb_lo_t2.json` — hold out t2
- `configs/singlecell/eb_dmsb_lo_t3.json` — hold out t3

**Run commands** (run one at a time to avoid resource contention):

```bash
# Train on all timepoints
python experiments/train.py --dataset singlecell --potential w2 \
    --config configs/singlecell/eb_dmsb_all.json --tag dmsb_all

# Leave-one-out
python experiments/train.py --dataset singlecell --potential w2 \
    --config configs/singlecell/eb_dmsb_lo_t1.json --tag dmsb_lo_t1

python experiments/train.py --dataset singlecell --potential w2 \
    --config configs/singlecell/eb_dmsb_lo_t2.json --tag dmsb_lo_t2

python experiments/train.py --dataset singlecell --potential w2 \
    --config configs/singlecell/eb_dmsb_lo_t3.json --tag dmsb_lo_t3
```

**Effective config** (from `defaults.json` + `W2.json` + experiment config):
- 100-dim PCA, normalized, OT-coupled
- 768d/8L architecture (4.9M params), adaptive loss
- W2Inf potential with OT coupling (= W2), strength 500
- 300 epochs, lr 0.003, batch size 256
- `tks` auto-computed: [0.25, 0.5, 0.75] for all, [0.33, 0.67] for LOO
- Metrics: MMD, SWD, FGD, W1, W2 at each timepoint

## Tutorial Notebooks

Interactive tutorials demonstrating each experiment:

1. `notebooks/01_quickstart_gaussians.ipynb` - Introduction with Gaussian data
2. `notebooks/02_singlecell_eb.ipynb` - Embryoid body single-cell trajectory inference
3. `notebooks/03_gulf_of_mexico.ipynb` - Ocean current modeling
4. `notebooks/04_beijing_airquality.ipynb` - Air quality forecasting
5. `notebooks/05_exact_gaussian_solutions.ipynb` - Analytical solutions
