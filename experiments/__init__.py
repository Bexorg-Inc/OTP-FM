"""
OTP-FM Experiments

This package contains code for reproducing experiments.
Each dataset has its own submodule with data loading, training, and evaluation.

Datasets:
    - gaussian: Synthetic Gaussian experiments
    - singlecell: Embryoid body single-cell RNA sequencing trajectory inference
    - gulfofmexico: Ocean current modeling in the Gulf of Mexico
    - beijingair: PM2.5 air quality forecasting

Common utilities:
    from experiments.evaluation import compute_fgd, compute_mmd, compute_swd
    from experiments.plotting import plot_losses_otp, plot_target_vs_learned
"""

import sys

from experiments.common import evaluation, plotting, trainer

# Register module aliases for cleaner imports
sys.modules["experiments.evaluation"] = evaluation
sys.modules["experiments.plotting"] = plotting
sys.modules["experiments.trainer"] = trainer
