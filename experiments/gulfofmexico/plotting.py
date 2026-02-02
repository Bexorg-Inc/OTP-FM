"""
Gulf of Mexico specific plotting utilities.

Author(s): Raghav Kansal
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.common.plotting import COLOURS


def plot_2d_trajectories(
    trajectories: torch.Tensor,
    time_points: np.ndarray,
    ground_truth_marginals: dict[int, torch.Tensor],
    num_trajectories: int = 111,
    title: str = "Ocean Current Trajectories",
    save_path: Path | None = None,
    show: bool = False,
):
    """
    Plot 2D trajectories with ground truth marginals.

    Args:
        trajectories: Sampled trajectories (n_steps, n_samples, 2)
        time_points: Time points for trajectories (n_steps,)
        ground_truth_marginals: Dict mapping time -> ground truth samples
        num_trajectories: Number of trajectories to plot
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color map for times
    all_times = sorted(ground_truth_marginals.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_times)))
    time_to_color = {t: colors[i] for i, t in enumerate(all_times)}

    # Panel 1: Ground truth marginals
    ax = axes[0]
    for t, samples in ground_truth_marginals.items():
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        color = time_to_color.get(t, "gray")
        ax.scatter(samples[:, 0], samples[:, 1], c=[color], alpha=0.5, s=3, label=f"t={t}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Ground Truth Marginals")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_aspect("equal")

    # Panel 2: Learned trajectories
    ax = axes[1]
    # Plot marginals as background
    for t, samples in ground_truth_marginals.items():
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        ax.scatter(samples[:, 0], samples[:, 1], c="lightgray", alpha=0.2, s=1)

    # Plot trajectories
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.cpu().numpy()
    n_plot = min(num_trajectories, trajectories.shape[1])
    for i in range(n_plot):
        ax.plot(
            trajectories[:, i, 0],
            trajectories[:, i, 1],
            alpha=0.5,
            linewidth=0.5,
            color=COLOURS["bexgreen"],
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Learned Trajectories")
    ax.set_aspect("equal")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_spatial_marginals(
    marginals: dict[int, torch.Tensor],
    title: str = "Spatial Marginals",
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot spatial marginal distributions at each time point."""
    n_times = len(marginals)
    ncols = min(5, n_times)
    nrows = (n_times + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    if n_times == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (t, samples) in enumerate(sorted(marginals.items())):
        ax = axes[idx]
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
        ax.set_title(f"t={t}")
        ax.set_aspect("equal")

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
