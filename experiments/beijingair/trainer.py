"""
Beijing air quality specific trainer.

Author(s): Raghav Kansal
"""

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from experiments.beijingair import plotting
from experiments.common.plotting import plot_target_vs_learned
from experiments.common.trainer import Trainer


class BeijingTrainer(Trainer):
    """Trainer for Beijing air quality PM2.5 experiments."""

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
        # Beijing-specific
        scaler=None,
        marginals: dict[int, Tensor] | None = None,
        train_times: list[int] | None = None,
        holdout_times: list[int] | None = None,
        eval_n_steps: int = 20,
        eval_num_samples: int = 1000,
        traj_skips: int | None = None,
    ):
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
        self.marginals = marginals
        self.train_times = train_times
        self.holdout_times = holdout_times
        self.eval_n_steps = eval_n_steps
        self.eval_num_samples = eval_num_samples

        if traj_skips is None:
            self.traj_skips = 1 if epochs <= 30 else max(1, math.ceil(epochs / 30))
        else:
            self.traj_skips = traj_skips

        self.epoch_trajectories: list[np.ndarray] = []
        self.trajectory_t_eval: np.ndarray | None = None

        self.xtk_plot_dir = self.save_dir / "xtk_plots"
        self.xtk_plot_dir.mkdir(parents=True, exist_ok=True)

    def _get_underlying_dataset(self):
        dataset = self.train_loader.dataset
        if hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        return dataset

    def on_train_start(self):
        self._save_epoch_trajectories(epoch=0)
        self._plot_xtk_comparison(epoch=0)

    def on_epoch_end(self, epoch: int, batch: Tensor | None = None):
        if ((epoch + 1) % self.traj_skips == 0) or (epoch == self.epochs - 1):
            self._save_epoch_trajectories(epoch=epoch + 1)
            self._plot_xtk_comparison(epoch=epoch + 1)

    @torch.no_grad()
    def _save_epoch_trajectories(self, epoch: int):
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

        traj_dir = self.save_dir / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            traj_dir / f"trajectories_epoch{epoch:04d}.npz",
            trajectories=trajectories.cpu().numpy(),
            t_eval=t_eval.cpu().numpy(),
        )

    @torch.no_grad()
    def _plot_xtk_comparison(self, epoch: int):
        if self.potentials is None or len(self.potentials) == 0:
            return

        dataset = self._get_underlying_dataset()
        if hasattr(dataset, "get_ot_aligned_samples"):
            try:
                batch = dataset.get_ot_aligned_samples(n_samples=5)
            except ValueError:
                batch = self._get_random_samples(5)
        else:
            batch = self._get_random_samples(5)

        otp_alpha = self.otp_alpha_func(epoch * len(self.train_loader))

        plot_target_vs_learned(
            model=self.model,
            batch=batch,
            potentials=self.potentials,
            otp_alpha=otp_alpha,
            n_samples=5,
            name=f"xtk_epoch{epoch:04d}",
            plot_dir=self.xtk_plot_dir,
            show=False,
            device=self.device,
        )

    def _get_random_samples(self, n_samples: int) -> torch.Tensor:
        samples = []
        for t in sorted(self.train_times):
            marginal = self.marginals[t]
            indices = np.random.choice(len(marginal), size=n_samples, replace=False)
            samples.append(marginal[indices].cpu())
        return torch.stack(samples, dim=1)

    def post_training(self, show: bool = False) -> Path:
        self.plot_losses(show=show)
        self.save_losses_csv()
        save_path = self.save_checkpoint("model.pt")

        if self.marginals is not None and self.epoch_trajectories:
            self._plot_final_trajectories()

        return save_path

    def _plot_final_trajectories(self):
        if not self.epoch_trajectories or self.trajectory_t_eval is None:
            return

        trajectories = torch.from_numpy(self.epoch_trajectories[-1])
        all_times = sorted(self.marginals.keys())
        gt_marginals = {t: self.marginals[t] for t in all_times}

        plotting.plot_1d_trajectories(
            trajectories=trajectories,
            time_points=self.trajectory_t_eval,
            ground_truth_marginals=gt_marginals,
            train_times=self.train_times,
            save_path=self.save_dir / "trajectories_1d.pdf",
            show=False,
        )
