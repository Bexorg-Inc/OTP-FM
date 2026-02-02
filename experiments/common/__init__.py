"""
Common utilities for OTP-FM experiments.

This module contains shared components for training, evaluation, and plotting
across all experiment types.
"""

from experiments.common.evaluation import (
    compute_fgd,
    compute_mmd,
    compute_swd,
    compute_w2_distance,
)
from experiments.common.plotting import (
    COLOURS,
    plot_losses_otp,
    plot_target_vs_learned,
    save_plot,
)
from experiments.common.trainer import Trainer, get_otp_alpha_func

__all__ = [
    # Trainer
    "Trainer",
    "get_otp_alpha_func",
    # Evaluation metrics
    "compute_swd",
    "compute_mmd",
    "compute_fgd",
    "compute_w2_distance",
    # Plotting
    "plot_target_vs_learned",
    "plot_losses_otp",
    "save_plot",
    "COLOURS",
]
