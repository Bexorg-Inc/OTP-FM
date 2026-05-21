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

**Config:** `configs/singlecell/5DL2O/eb_ijko.json` (layers on top of `defaults.json` → `W2.json`). Includes `consistency_loss=imf` since Phase-2 5-seed validation (Avg W2 over (t1, t3) = 0.8268 +/- 0.0030) showed IMF beats the meanflow default (0.8331 single-seed) on this experiment.

**Run command:**

```bash
python experiments/train.py --dataset singlecell --potential w2 \
    --config configs/singlecell/5DL2O/eb_ijko.json --tag ijko_w2
```

**Notes:**
- W2 is computed in normalized (standardized) space, consistent with the iJKOnet paper and previous benchmarks.

### EB 100D Leave-One-Out (DMSB Paper Comparison)

This reproduces the Embryoid Body experiment from [Chen et al. 2023 "Deep Multi-Marginal Momentum Schrödinger Bridge"](https://arxiv.org/abs/2303.01751), Table 3. The experiment uses **100-dim PCA** with the standard single-cell config (`defaults.json`), evaluating MMD, SWD, and FGD at each timepoint. Four conditions: train-on-all, and leave-out t1/t2/t3. Each condition is run with W2 (OT-coupled) and W2Inf (no OT coupling).

**Configs:** Self-contained configs in `configs/singlecell/100D/` layer on top of `defaults.json`. Common overrides shared by all conditions are documented in `100D/defaults.json` (MSE loss, strength 300). LOO t1 and t3 use wider potentials (`width=0.3`), more epochs (500), a faster alpha schedule (`otp_alpha_mean_scale=0.3`), and gradient clipping (`grad_clip=1.0`).

**Run commands** (run one at a time to avoid GPU contention):

```bash
# Train on all timepoints
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/all_w2.json --tag 100d_all_w2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/all_w2inf.json --tag 100d_all_w2inf

# Leave-one-out: t1
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t1_w2.json --tag 100d_lo_t1_w2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t1_w2inf.json --tag 100d_lo_t1_w2inf

# Leave-one-out: t2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t2_w2.json --tag 100d_lo_t2_w2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t2_w2inf.json --tag 100d_lo_t2_w2inf

# Leave-one-out: t3
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t3_w2.json --tag 100d_lo_t3_w2
python experiments/train.py --dataset singlecell \
    --config configs/singlecell/100D/lo_t3_w2inf.json --tag 100d_lo_t3_w2inf
```

**Effective config** (from `defaults.json` + experiment config):
- 100-dim PCA, normalized, 768d/8L architecture (4.9M params)
- MSE loss, W2Inf potential, strength 300
- `tks` auto-computed: [0.25, 0.5, 0.75] for all, [0.33, 0.67] for LOO
- Train-on-all / LOO t2: 300 epochs, lr 0.003, batch size 256, width 0.2
- LOO t1 / LOO t3: 500 epochs, width 0.3, `otp_alpha_mean_scale` 0.3, `grad_clip` 1.0
- Metrics: MMD, SWD, FGD, W1 at each timepoint

**Expected results** (average MMD across t1–t4, best epoch):

| Condition | W2 (OT) | W2Inf (no OT) |
|-----------|:-------:|:-------------:|
| **All** | **0.035** | 0.066 |
| **LOO t1** | 0.070 (t1\*≈0.18) | 0.075 (t1\*≈0.18) |
| **LOO t2** | 0.057 (t2\*≈0.07) | 0.068 (t2\*≈0.14) |
| **LOO t3** | 0.067 (t3\*≈0.07) | 0.091 (t3\*≈0.10) |

**Notes:**
- W2 = W2Inf potential with OT-coupled sampling; W2Inf = W2Inf potential with random coupling.
- Train-on-all with W2 meets the <0.04 average MMD target. LOO experiments remain above 0.04, with LOO t1 being the hardest (held-out t1 MMD ≈ 0.18).
- LOO t1/t3 benefit from wider potentials and slower alpha ramp-up to stabilize training.

### CITE-seq 50D PCA (Leave-One-Out)

This experiment evaluates OTP-FM on the CITE-seq dataset (31,240 cells, 4 timepoints: days 2/3/4/7) using 50 PCA dimensions, following the protocol from [Neklyudov et al. 2024](https://arxiv.org/abs/2310.10649). Leave-one-out cross-validation holds out day 3 (fold 1) or day 4 (fold 2), and the primary metric is Wasserstein-1 distance in original PCA space.

**Data**: Download `cite_pca50.csv` from the [VGFM repository](https://github.com/DongyiWang-66/VGFM/blob/main/data/cite_pca50.csv) and place it in `OTP-FM/data/cite_pca50.csv`.

**Architecture**: 10-layer / 768-dim MLP with SiLU activation, LayerNorm, and residual connections every 2 layers (~6.5M params).

**Key hyperparameters**: W2Inf potential with delta kernel (strength=300, width=0.33), adaptive loss, meanflow consistency loss, cosine LR schedule (lr=0.003) over 200 epochs.

**Run commands:**

```bash
# W2 (OT-coupled, 200 epochs, cosine LR)
python experiments/train.py --dataset citeseq --potential W2 --holdout-times 1  # fold 1: hold out day 3
python experiments/train.py --dataset citeseq --potential W2 --holdout-times 2  # fold 2: hold out day 4

# W2Inf (no OT coupling, 80 epochs, no LR schedule)
python experiments/train.py --dataset citeseq --potential W2Inf --holdout-times 1
python experiments/train.py --dataset citeseq --potential W2Inf --holdout-times 2
```

**Expected results** (W1 on held-out marginal, original PCA space):

| Method | Fold 1 (t1/day 3) | Fold 2 (t2/day 4) | Average |
|--------|:-----------------:|:-----------------:|:-------:|
| **OTP-FM W2** | **38.07** (@ep70) | **35.46** (@ep10) | **36.76** |
| OTP-FM W2Inf | 41.99 (@ep80) | 33.28 (@ep80) | 37.64 |

**Notes:**
- Configs layer on `OTP-FM/configs/citeseq/defaults.json`; `W2.json` and `W2Inf.json` specify only the method-specific overrides.
- Holdout fold is selected via `--holdout-times` on the CLI — no separate config per fold.
- All metrics are computed on the full dataset (no subsampling) in the original (un-normalized) PCA space.
- W2 fold 2 peaks very early (~epoch 10) then overfits; fold 1 peaks around epoch 70.
- The delta potential shape and cosine LR schedule were identified as key improvements over the Gaussian potential baseline.
- `tks=[0.5]` is auto-computed for leave-one-out with one intermediate marginal.

## Tutorial Notebooks

Interactive tutorials demonstrating each experiment:

1. `notebooks/01_quickstart_gaussians.ipynb` - Introduction with Gaussian data
2. `notebooks/02_singlecell_eb.ipynb` - Embryoid body single-cell trajectory inference
3. `notebooks/03_gulf_of_mexico.ipynb` - Ocean current modeling
4. `notebooks/04_beijing_airquality.ipynb` - Air quality forecasting
5. `notebooks/05_exact_gaussian_solutions.ipynb` - Analytical solutions
