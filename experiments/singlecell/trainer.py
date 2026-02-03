"""
Single-cell specific trainer for Embryoid Body (EB) data.

Author(s): Raghav Kansal
"""

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from experiments import Trainer
from experiments.evaluation import compute_fgd, compute_swd, compute_w2_distance
from experiments.plotting import plot_target_vs_learned
from experiments.singlecell import plotting


class EBTrainer(Trainer):
    """
    Trainer for Embryoid Body (EB) single-cell experiments.

    Extends the base Trainer with:
    - Reshuffling cell pairings each epoch
    - Automatic evaluation on held-out time points
    - Single-cell specific visualizations
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Path,
        lr: float,
        # Training parameters
        epochs: int = 10,
        optimizer: str = "adam",
        grad_clip: float = 0.0,
        do_otp: bool = True,
        # Progressive loss weighting
        otp_alpha_type: str = "sigmoid",
        otp_alpha_slope: float = 6.0,
        otp_alpha_mean_scale: float = 1.0,
        # Sampling/evaluation
        sampling_steps: int = 50,
        ema_eval: bool = True,
        # Model
        potentials: OrderedDict | None = None,
        device: str = "cpu",
        # EB-specific
        scaler=None,
        reshuffle_each_epoch: bool = True,
        marginals: dict[int, Tensor] | None = None,
        train_times: list[int] | None = None,
        holdout_times: list[int] | None = None,
        eval_n_steps: int = 50,
        eval_num_samples: int = 2000,
        traj_skips: int | None = None,
        num_1d_plot_samples: int = 5,
    ):
        """Initialize the EB trainer."""
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            lr=lr,
            epochs=epochs,
            optimizer=optimizer,
            grad_clip=grad_clip,
            do_otp=do_otp,
            otp_alpha_type=otp_alpha_type,
            otp_alpha_slope=otp_alpha_slope,
            otp_alpha_mean_scale=otp_alpha_mean_scale,
            sampling_steps=sampling_steps,
            ema_eval=ema_eval,
            potentials=potentials,
            device=device,
        )

        self.scaler = scaler
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.marginals = marginals
        self.train_times = train_times
        self.holdout_times = holdout_times
        self.eval_n_steps = eval_n_steps
        self.eval_num_samples = eval_num_samples
        self.num_1d_plot_samples = num_1d_plot_samples

        # Auto-compute traj_skips
        if traj_skips is None:
            self.traj_skips = 1 if epochs <= 30 else max(1, math.ceil(epochs / 30))
        else:
            self.traj_skips = traj_skips

        # Trajectory storage
        self.epoch_trajectories: list[np.ndarray] = []
        self.trajectory_t_eval: np.ndarray | None = None

        # Plotting directories
        self.xtk_plot_dir = self.save_dir / "xtk_plots"
        self.xtk_plot_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metric tracking
        self._init_marginal_metrics()

    def _get_underlying_dataset(self):
        """Get the underlying dataset, handling random_split wrapper."""
        dataset = self.train_loader.dataset
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def _init_marginal_metrics(self):
        """Initialize metric tracking."""
        metrics = ["swd", "mmd", "fgd", "w2"]
        times = ["t1", "t2", "t3", "t4", "t2_t4"]
        for m in metrics:
            for t in times:
                self.losses[f"{m}_{t}"] = []
        self.losses["metric_epochs"] = []

    def on_train_start(self):
        """Save initial state before training."""
        self._save_epoch_trajectories(epoch=0)
        self._plot_xtk_comparison(epoch=0)
        self._compute_marginal_metrics(epoch=0)

    def on_epoch_start(self, epoch: int, batch: Tensor | None = None):
        """Reshuffle cell pairings."""
        if self.reshuffle_each_epoch:
            dataset = self._get_underlying_dataset()
            if hasattr(dataset, "reshuffle"):
                dataset.reshuffle()

    def on_epoch_end(self, epoch: int, batch: Tensor | None = None):
        """Save trajectories and compute metrics."""
        if ((epoch + 1) % self.traj_skips == 0) or (epoch == self.epochs - 1):
            self._save_epoch_trajectories(epoch=epoch + 1)
            self._plot_xtk_comparison(epoch=epoch + 1)
            if epoch != self.epochs - 1:
                self._compute_marginal_metrics(epoch=epoch + 1, compute_mmd=False)

    @torch.no_grad()
    def _save_epoch_trajectories(self, epoch: int):
        """Sample and save trajectories."""
        if self.marginals is None or self.train_times is None:
            return

        self.model.eval()
        source_time = min(self.train_times)
        source = self.marginals[source_time]
        num_samples = min(self.eval_num_samples, len(source))
        x0 = source[:num_samples].to(self.device)

        trajectories, t_eval = self.model.sample(x0, n_steps=self.eval_n_steps, ema=self.ema_eval)

        self.epoch_trajectories.append(trajectories.cpu().numpy())
        if self.trajectory_t_eval is None:
            self.trajectory_t_eval = t_eval.cpu().numpy()

        # Save to disk
        traj_dir = self.save_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            traj_dir / f"trajectories_epoch{epoch:04d}.npz",
            trajectories=trajectories.cpu().numpy(),
            t_eval=t_eval.cpu().numpy(),
        )

    @torch.no_grad()
    def _plot_xtk_comparison(self, epoch: int):
        """Plot X_tk comparison."""
        if self.potentials is None or len(self.potentials) == 0:
            return

        n_samples = self.num_1d_plot_samples
        dataset = self._get_underlying_dataset()

        if hasattr(dataset, "get_ot_aligned_samples"):
            try:
                batch = dataset.get_ot_aligned_samples(n_samples=n_samples)
            except ValueError:
                batch = self._get_random_samples(n_samples)
        else:
            batch = self._get_random_samples(n_samples)

        otp_alpha = self.otp_alpha_func(epoch * len(self.train_loader))

        plot_target_vs_learned(
            model=self.model,
            batch=batch,
            potentials=self.potentials,
            otp_alpha=otp_alpha,
            n_samples=n_samples,
            name=f"xtk_epoch{epoch:04d}",
            plot_dir=self.xtk_plot_dir,
            show=False,
            device=self.device,
        )

    def _get_random_samples(self, n_samples: int) -> torch.Tensor:
        """Get random samples from each marginal."""
        samples = []
        for t in sorted(self.train_times):
            marginal = self.marginals[t]
            indices = np.random.choice(len(marginal), size=n_samples, replace=False)
            samples.append(marginal[indices].cpu())
        return torch.stack(samples, dim=1)  # (n_samples, n_times, dim)

    @torch.no_grad()
    def _compute_marginal_metrics(self, epoch: int, compute_mmd: bool = False):
        """Compute metrics at all time points."""
        if self.marginals is None or not self.epoch_trajectories:
            return

        trajectories = self.epoch_trajectories[-1]
        if self.scaler is not None:
            # Inverse transform if normalized
            shape = trajectories.shape
            trajectories = self.scaler.inverse_transform(
                trajectories.reshape(-1, shape[-1])
            ).reshape(shape)

        t_eval = self.trajectory_t_eval

        # Time normalization
        time_min, time_max = min(self.train_times), max(self.train_times)

        def normalize_time(t):
            return (t - time_min) / (time_max - time_min)

        generated_samples = {}

        for t in [1, 2, 3, 4]:
            if t not in self.marginals:
                continue

            t_norm = normalize_time(t)
            target_idx = np.argmin(np.abs(t_eval - t_norm))

            ground_truth = self.marginals[t][: self.eval_num_samples]
            if self.scaler is not None:
                ground_truth = torch.from_numpy(
                    self.scaler.inverse_transform(ground_truth.cpu().numpy())
                )
            generated = torch.from_numpy(trajectories[target_idx])

            # Compute metrics
            self.losses[f"swd_t{t}"].append(compute_swd(generated, ground_truth))
            if compute_mmd:
                self.losses[f"mmd_t{t}"].append(compute_mmd(generated, ground_truth))
            self.losses[f"fgd_t{t}"].append(compute_fgd(generated, ground_truth))
            w2_dim = min(generated.shape[-1], 10)
            self.losses[f"w2_t{t}"].append(
                compute_w2_distance(generated[:, :w2_dim], ground_truth[:, :w2_dim])
            )

            generated_samples[t] = generated

        # Combined t2+t4 metrics
        if 2 in generated_samples and 4 in generated_samples:
            gen_combined = torch.cat([generated_samples[2], generated_samples[4]])
            gt_2 = self.marginals[2][: self.eval_num_samples]
            gt_4 = self.marginals[4][: self.eval_num_samples]
            if self.scaler is not None:
                gt_2 = torch.from_numpy(self.scaler.inverse_transform(gt_2.cpu().numpy()))
                gt_4 = torch.from_numpy(self.scaler.inverse_transform(gt_4.cpu().numpy()))
            gt_combined = torch.cat([gt_2, gt_4])

            self.losses["swd_t2_t4"].append(compute_swd(gen_combined, gt_combined))
            if compute_mmd:
                self.losses["mmd_t2_t4"].append(compute_mmd(gen_combined, gt_combined))
            self.losses["fgd_t2_t4"].append(compute_fgd(gen_combined, gt_combined))

        self.losses["metric_epochs"].append(epoch)

    def post_training(self, show: bool = False) -> Path:
        """Run post-training tasks."""
        self._compute_marginal_metrics(epoch=self.epochs, compute_mmd=True)

        # Plot losses
        self.plot_losses(show=show)
        self.save_losses_csv()

        # Save final model
        save_path = self.save_checkpoint("model.pt")
        self.logger.info(f"Model saved to {save_path}")

        # Plot trajectories
        if self.marginals is not None and self.epoch_trajectories:
            self._plot_final_trajectories()

        return save_path

    def _plot_final_trajectories(self):
        """Plot final trajectory visualization."""
        if not self.epoch_trajectories or self.trajectory_t_eval is None:
            return

        trajectories = torch.from_numpy(self.epoch_trajectories[-1])
        all_times = sorted(self.marginals.keys())
        gt_marginals = {t: self.marginals[t] for t in all_times}

        plotting.plot_pca_trajectories(
            trajectories=trajectories,
            time_points=self.trajectory_t_eval,
            ground_truth_marginals=gt_marginals,
            plot_times=all_times,
            pcs=(0, 1),
            save_path=self.save_dir / "trajectories_pc1_pc2.pdf",
            show=False,
        )
