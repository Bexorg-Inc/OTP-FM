"""
Single-cell specific plotting utilities.

Author(s): Raghav Kansal
"""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from experiments.plotting import COLOURS, save_plot


def plot_pca_trajectories(
    trajectories: torch.Tensor,
    time_points: np.ndarray,
    ground_truth_marginals: dict[int, torch.Tensor],
    plot_times: list[int],
    pcs: tuple[int, int] = (0, 1),
    num_trajectories: int = 100,
    title: str = "Cell Trajectories",
    save_path: Path | None = None,
    show: bool = False,
    ot_samples: np.ndarray | None = None,
    ot_times: list[int] | None = None,
):
    """
    Plot trajectories in PCA space with ground truth marginals.

    Args:
        trajectories: Sampled trajectories (n_steps, n_samples, dim)
        time_points: Time points for trajectories (n_steps,)
        ground_truth_marginals: Dict mapping time -> ground truth samples
        plot_times: List of times for color mapping
        pcs: Tuple of (pc1_idx, pc2_idx) to plot
        num_trajectories: Number of trajectories to plot
        title: Plot title
        save_path: Path to save figure
        show: Whether to display figure
        ot_samples: Optional OT-aligned samples (n_samples, n_times, dim)
        ot_times: Times corresponding to ot_samples
    """
    pc1, pc2 = pcs
    n_panels = 3 if ot_samples is not None else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    # Color map for times
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_times)))
    time_to_color = {t: colors[i] for i, t in enumerate(plot_times)}

    # Panel 1: Ground truth marginals
    ax = axes[0]
    for t, samples in ground_truth_marginals.items():
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        color = time_to_color.get(t, "gray")
        ax.scatter(samples[:, pc1], samples[:, pc2], c=[color], alpha=0.3, s=1, label=f"t={t}")
    ax.set_xlabel(f"PC{pc1 + 1}")
    ax.set_ylabel(f"PC{pc2 + 1}")
    ax.set_title("Ground Truth Marginals")
    ax.legend(markerscale=5, fontsize=8)

    # Panel 2: OT-coupled trajectories (if available)
    if ot_samples is not None and n_panels > 2:
        ax = axes[1]
        # Plot marginals as background
        for t, samples in ground_truth_marginals.items():
            if isinstance(samples, torch.Tensor):
                samples = samples.cpu().numpy()
            ax.scatter(samples[:, pc1], samples[:, pc2], c="lightgray", alpha=0.1, s=1)

        # Plot OT trajectories
        n_plot = min(num_trajectories, len(ot_samples))
        for i in range(n_plot):
            ax.plot(
                ot_samples[i, :, pc1],
                ot_samples[i, :, pc2],
                alpha=0.5,
                linewidth=0.5,
                color=COLOURS["bexpurple"],
            )
        ax.set_xlabel(f"PC{pc1 + 1}")
        ax.set_ylabel(f"PC{pc2 + 1}")
        ax.set_title("OT-Coupled Trajectories")

    # Final panel: Learned trajectories
    ax = axes[-1]
    # Plot marginals as background
    for t, samples in ground_truth_marginals.items():
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        ax.scatter(samples[:, pc1], samples[:, pc2], c="lightgray", alpha=0.1, s=1)

    # Plot learned trajectories
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.cpu().numpy()
    n_plot = min(num_trajectories, trajectories.shape[1])
    for i in range(n_plot):
        ax.plot(
            trajectories[:, i, pc1],
            trajectories[:, i, pc2],
            alpha=0.5,
            linewidth=0.5,
            color=COLOURS["bexgreen"],
        )
    ax.set_xlabel(f"PC{pc1 + 1}")
    ax.set_ylabel(f"PC{pc2 + 1}")
    ax.set_title("Learned Trajectories")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def plot_losses(
    losses: dict,
    name: str = "losses",
    plot_dir: Path | None = None,
    log: bool = False,
    show: bool = False,
):
    """
    Plot training losses and metrics for single-cell experiments.

    Args:
        losses: Dictionary of loss values
        name: Base name for saved file
        plot_dir: Directory to save plot
        log: Whether to use log scale
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Training loss
    ax = axes[0, 0]
    if "train_loss" in losses and losses["train_loss"]:
        train_epochs = np.arange(1, len(losses["train_loss"]) + 1)
        if log:
            ax.semilogy(
                train_epochs, losses["train_loss"], label="Train", color=COLOURS["brightorange"]
            )
        else:
            ax.plot(
                train_epochs, losses["train_loss"], label="Train", color=COLOURS["brightorange"]
            )

    if "val_loss" in losses and losses["val_loss"]:
        val_epochs = np.arange(0, len(losses["val_loss"]))
        if log:
            ax.semilogy(
                val_epochs, losses["val_loss"], "--", label="Val", color=COLOURS["brightorange"]
            )
        else:
            ax.plot(
                val_epochs, losses["val_loss"], "--", label="Val", color=COLOURS["brightorange"]
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    # Panel 2: OTP Alpha
    ax = axes[0, 1]
    if "otp_alpha" in losses and losses["otp_alpha"]:
        otp_alpha = np.array(losses["otp_alpha"])
        ax.plot(otp_alpha[:, 0], otp_alpha[:, 1], color=COLOURS["bexpurple"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\alpha(i)$")
        ax.set_title("OTP Alpha Schedule")
        ax.set_ylim(0, 1)

    # Panel 3: SWD metrics
    ax = axes[1, 0]
    metric_epochs = losses.get("metric_epochs", [])
    if metric_epochs:
        for t in [1, 2, 3, 4]:
            key = f"swd_t{t}"
            if key in losses and losses[key]:
                ax.plot(metric_epochs[: len(losses[key])], losses[key], label=f"t{t}")
        if "swd_t2_t4" in losses and losses["swd_t2_t4"]:
            ax.plot(
                metric_epochs[: len(losses["swd_t2_t4"])], losses["swd_t2_t4"], "--", label="t2+t4"
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SWD")
    ax.set_title("Sliced Wasserstein Distance")
    ax.legend()

    # Panel 4: FGD metrics
    ax = axes[1, 1]
    if metric_epochs:
        for t in [1, 2, 3, 4]:
            key = f"fgd_t{t}"
            if key in losses and losses[key]:
                ax.plot(metric_epochs[: len(losses[key])], losses[key], label=f"t{t}")
        if "fgd_t2_t4" in losses and losses["fgd_t2_t4"]:
            ax.plot(
                metric_epochs[: len(losses["fgd_t2_t4"])], losses["fgd_t2_t4"], "--", label="t2+t4"
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FGD")
    ax.set_title("Fréchet Gaussian Distance")
    ax.legend()

    plt.tight_layout()
    save_plot(plot_dir, name, show)


def create_trajectory_animation(
    epoch_trajectories: list[np.ndarray],
    ground_truth_marginals: dict[int, torch.Tensor],
    trajectory_t_eval: np.ndarray,
    save_path: Path,
    traj_skips: int = 1,
    num_trajectories: int = 100,
    pcs: tuple[int, int] = (0, 1),
    duration: int = 500,
):
    """
    Create animated GIF of trajectory evolution across epochs.

    Args:
        epoch_trajectories: List of trajectory arrays (n_steps, n_samples, dim)
        ground_truth_marginals: Dict of ground truth samples per time
        trajectory_t_eval: Time points for trajectories
        save_path: Path to save GIF
        traj_skips: Number of epochs between saves
        num_trajectories: Number of trajectories to show
        pcs: PC indices to plot
        duration: Frame duration in ms
    """
    frames = []
    pc1, pc2 = pcs

    for epoch_idx, trajectories in enumerate(tqdm(epoch_trajectories, desc="Creating animation")):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot ground truth as background
        for t, samples in ground_truth_marginals.items():
            if isinstance(samples, torch.Tensor):
                samples = samples.cpu().numpy()
            ax.scatter(samples[:, pc1], samples[:, pc2], c="lightgray", alpha=0.1, s=1)

        # Plot trajectories
        n_plot = min(num_trajectories, trajectories.shape[1])
        for i in range(n_plot):
            ax.plot(
                trajectories[:, i, pc1],
                trajectories[:, i, pc2],
                alpha=0.5,
                linewidth=0.5,
                color=COLOURS["bexgreen"],
            )

        ax.set_xlabel(f"PC{pc1 + 1}")
        ax.set_ylabel(f"PC{pc2 + 1}")
        ax.set_title(f"Epoch {epoch_idx * traj_skips}")

        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img.copy())
        plt.close(fig)
        buf.close()

    if frames:
        durations = [duration * 5] + [duration] * (len(frames) - 2) + [duration * 10]
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
        )
