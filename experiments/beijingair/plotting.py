"""
Beijing air quality specific plotting utilities.

Author(s): Raghav Kansal
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.plotting import COLOURS


def plot_1d_trajectories(
    trajectories: torch.Tensor,
    time_points: np.ndarray,
    ground_truth_marginals: dict[int, torch.Tensor],
    train_times: list[int],
    num_trajectories: int = 200,
    title: str = "PM2.5 Trajectories",
    save_path: Path | None = None,
    show: bool = False,
):
    """
    Plot 1D trajectories with ground truth marginals.

    Args:
        trajectories: Sampled trajectories (n_steps, n_samples, 1)
        time_points: Normalized time points (n_steps,)
        ground_truth_marginals: Dict mapping time -> ground truth samples
        train_times: Training time indices
        num_trajectories: Number of trajectories to plot
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_times = sorted(ground_truth_marginals.keys())
    time_min, time_max = min(all_times), max(all_times)

    def normalize_time(t):
        return (t - time_min) / (time_max - time_min)

    # Panel 1: Ground truth marginals as violin plots
    ax = axes[0]
    positions = [normalize_time(t) for t in all_times]
    data = []
    for t in all_times:
        samples = ground_truth_marginals[t]
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        data.append(samples.flatten())

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False)

    # Color by train/holdout
    for i, t in enumerate(all_times):
        color = COLOURS["bexgreen"] if t in train_times else COLOURS["brightorange"]
        if "bodies" in parts:
            parts["bodies"][i].set_facecolor(color)
            parts["bodies"][i].set_alpha(0.6)

    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("PM2.5")
    ax.set_title("Ground Truth Marginals (green=train, orange=holdout)")

    # Panel 2: Learned trajectories
    ax = axes[1]

    # Plot trajectories
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.cpu().numpy()
    n_plot = min(num_trajectories, trajectories.shape[1])
    for i in range(n_plot):
        ax.plot(
            time_points,
            trajectories[:, i, 0],
            alpha=0.3,
            linewidth=0.5,
            color=COLOURS["bexgreen"],
        )

    # Add violin plots on top
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=False)
    for i, t in enumerate(all_times):
        color = COLOURS["bexgreen"] if t in train_times else COLOURS["brightorange"]
        if "bodies" in parts:
            parts["bodies"][i].set_facecolor(color)
            parts["bodies"][i].set_alpha(0.4)

    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("PM2.5")
    ax.set_title("Learned Trajectories")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_pm25_distributions(
    marginals: dict[int, torch.Tensor],
    train_times: list[int],
    title: str = "PM2.5 Distributions",
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot PM2.5 distributions at each time point as histograms."""
    n_times = len(marginals)
    ncols = min(4, n_times)
    nrows = (n_times + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if n_times == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (t, samples) in enumerate(sorted(marginals.items())):
        ax = axes[idx]
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        color = COLOURS["bexgreen"] if t in train_times else COLOURS["brightorange"]
        ax.hist(samples.flatten(), bins=30, alpha=0.7, color=color)
        ax.set_title(f"t={t}")
        ax.set_xlabel("PM2.5")

    # Hide empty axes
    for idx in range(len(marginals), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
