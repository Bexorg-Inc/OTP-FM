"""
Base Trainer class for OTP-FM experiments.

Author(s): Raghav Kansal
"""

import gc
import logging
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from otpfm import Curriculum
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments import plotting

logger = logging.getLogger(__name__)


class Trainer:
    """
    Base trainer for OTP-FM.

    This class provides core training functions and utilities without domain-specific features,
    and subclass and override hooks for dataset-specific behavior.
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
        sampling_method: str = "consistency",
        ema_eval: bool = True,
        # Model
        potentials: OrderedDict | None = None,
        device: str = "cpu",
        traj_skips: int | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: The OTP-FM model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            save_dir: Directory to save checkpoints and plots
            lr: Learning rate
            epochs: Number of training epochs
            optimizer: Optimizer type ("adam", "sgd", "rmsprop")
            grad_clip: Gradient clipping norm (0 = disabled)
            do_otp: Whether to apply OTP corrections during training
            otp_alpha_type: Progressive loss weight type ("sigmoid", "0", "1")
            otp_alpha_slope: Slope for sigmoid schedule
            otp_alpha_mean_scale: Mean scale for sigmoid schedule (default: 1.0)
            sampling_steps: Number of steps for trajectory sampling
            sampling_method: Sampling method ("consistency" or "ode")
            ema_eval: Whether to use EMA model for evaluation
            potentials: OrderedDict mapping tk -> Potential. If None, uses model.potentials.
            device: Device to train on ("cpu" or "cuda")
            traj_skips: How often to save checkpoints (default: auto based on epochs)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.lr = lr
        self.device = device

        # Training parameters
        self.epochs = epochs
        self.optimizer_name = optimizer
        self.grad_clip = grad_clip
        self.do_otp = do_otp

        # Sampling/evaluation
        self.sampling_steps = sampling_steps
        self.sampling_method = sampling_method
        self.ema_eval = ema_eval

        # Use provided potentials or get from model
        self.potentials = (
            potentials if potentials is not None else getattr(model, "potentials", None)
        )

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logger
        self._setup_file_logging()
        self.logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Save directory: {self.save_dir}")
        self.logger.info(f"Epochs: {epochs}, LR: {lr}, Optimizer: {optimizer}")
        if self.potentials:
            self.logger.info(f"Potentials: {list(self.potentials.keys())}")

        # Models directory
        self.models_dir = self.save_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Set up optimizer
        self.optimizers = {}
        self._setup_optimizer(optimizer, lr)

        # Initialize loss tracking
        self.losses: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "otp_alpha": [],
        }

        # Global step counter
        self.global_step = 0

        # Curriculum for progressive loss weighting
        self.total_steps = epochs * len(train_loader)
        self.curriculum = Curriculum(
            total_iterations=self.total_steps,
            schedule=otp_alpha_type,
            slope=2 * otp_alpha_slope,  # need to scale to match with old definitions
            midpoint=0.5 * otp_alpha_mean_scale,
        )

        # Animation settings
        if traj_skips is None:
            if epochs <= 30:
                self.traj_skips = 1
            else:
                self.traj_skips = min(epochs // 20, 50)
        else:
            self.traj_skips = traj_skips

    def _setup_file_logging(self) -> None:
        """Set up file handler for logging."""
        log_file = self.save_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.log_file = log_file

    def _setup_optimizer(self, optimizer: str, lr: float) -> None:
        """Set up the optimizer."""
        match optimizer:
            case "adam":
                self.optimizers["flow"] = torch.optim.Adam(self.model.flownet.parameters(), lr=lr)
            case "sgd":
                self.optimizers["flow"] = torch.optim.SGD(
                    self.model.flownet.parameters(), lr=lr, momentum=0.9
                )
            case "rmsprop":
                self.optimizers["flow"] = torch.optim.RMSprop(
                    self.model.flownet.parameters(), lr=lr
                )
            case _:
                raise ValueError(f"Invalid optimizer: {optimizer}")

    def train_epoch(self, epoch: int) -> tuple[float, Tensor]:
        """
        Run one training epoch.

        Returns:
            Loss averaged over epoch, batch for debugging purposes
        """
        self.model.train()

        epoch_loss = 0.0
        ret_batch = None
        slow_threshold = 2.0  # seconds — log batches slower than this

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            batch_t0 = time.perf_counter()

            # Zero gradients
            for opt in self.optimizers.values():
                opt.zero_grad()

            batch = self._process_batch(batch).to(self.device)

            # Forward pass
            loss = self.model.forward_with_loss(
                batch,
                self.curriculum(self.global_step),
                do_otp=self.do_otp,
            )

            # Backward pass with progressive weighting
            otp_alpha = self.curriculum(self.global_step)

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)

            loss.backward()
            self.optimizers["flow"].step()

            self.model.update_ema()

            # Accumulate losses
            epoch_loss += self._get_item(loss)

            # Track progressive loss weight
            self.losses["otp_alpha"].append([self.global_step / len(self.train_loader), otp_alpha])

            self.global_step += 1

            # Detect slow batches
            batch_dt = time.perf_counter() - batch_t0
            if batch_dt > slow_threshold:
                self.logger.warning(
                    f"Slow batch: epoch {epoch}, batch {i}/{len(self.train_loader)}, "
                    f"took {batch_dt:.1f}s (GPU mem: "
                    f"{torch.cuda.memory_allocated() / 1e9:.2f}GB allocated, "
                    f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved)"
                )

            # Update progress bar
            pbar.set_postfix({"loss": f"{self._get_item(loss):.4f}"})

            if i == len(self.train_loader) - 2:
                # Returns second-last batch (so batch is full sized)
                ret_batch = batch

        n = len(self.train_loader)
        return epoch_loss / n, ret_batch

    @torch.no_grad()
    def validate(self) -> tuple[float, Tensor]:
        """
        Run validation.

        Returns:
            Loss averaged over validation set, batch for debugging purposes
        """
        self.model.eval()

        val_loss = 0.0
        ret_batch = None

        for i, batch in enumerate(self.val_loader):
            batch = self._process_batch(batch).to(self.device)

            loss = self.model.forward_with_loss(
                batch,
                self.curriculum(self.global_step),
                do_otp=self.do_otp,
                debug=(i == 0),
            )

            val_loss += self._get_item(loss)

            if i == len(self.val_loader) - 2:
                # Returns second-last batch (so batch is full sized)
                ret_batch = batch

        n = len(self.val_loader)

        val_loss = (val_loss / n) if n > 0 else 0.0
        return val_loss, ret_batch

    def save_checkpoint(self, name: str = "model.pt") -> Path:
        """Save model checkpoint."""
        save_path = self.models_dir / name
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "losses": self.losses,
                "epochs": self.epochs,
                "global_step": self.global_step,
            },
            save_path,
        )
        return save_path

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.losses = checkpoint.get("losses", self.losses)
        self.global_step = checkpoint.get("global_step", 0)

    def on_epoch_start(self, epoch: int, batch: Tensor | None = None) -> None:
        """Hook called at the start of each epoch. Override in subclasses."""
        pass

    def on_epoch_end(self, epoch: int, batch: Tensor | None = None) -> None:
        """Hook called at the end of each epoch. Override in subclasses."""
        pass

    def on_train_start(self) -> None:
        """Hook called before training starts. Override in subclasses."""
        pass

    def on_train_end(self) -> None:
        """Hook called after training completes. Override in subclasses."""
        pass

    def train(self) -> tuple[dict[str, list[float]], Tensor | None]:
        """
        Run the full training loop.

        Returns:
            Tuple of (losses dict, last batch)
        """
        self.on_train_start()

        # Disable automatic GC during training to prevent mid-epoch pauses;
        # we run it explicitly between epochs instead.
        gc.disable()

        # Initial validation
        val_loss, batch = self.validate()
        self.losses["val_loss"].append(val_loss)

        for epoch in range(self.epochs):
            self.on_epoch_start(epoch, batch)

            # Explicit GC between epochs (not during training)
            gc.collect()
            torch.cuda.empty_cache()

            # Training
            train_loss, batch = self.train_epoch(epoch)
            self.losses["train_loss"].append(train_loss)

            # Validation
            val_loss, _ = self.validate()
            self.losses["val_loss"].append(val_loss)

            if (((epoch + 1) % self.traj_skips) == 0) or (epoch == self.epochs - 1):
                self.save_checkpoint(f"model_epoch_{epoch + 1:03d}.pt")
                self.on_epoch_end(epoch, batch)

            # Log epoch summary
            self.logger.info(
                f"Epoch {epoch}: " f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        # Re-enable GC after training
        gc.enable()
        gc.collect()

        self.on_train_end()

        self.logger.info(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(
            f"Final losses: train_loss={self.losses['train_loss'][-1]:.4f}, "
            f"val_loss={self.losses['val_loss'][-1]:.4f}"
        )
        return self.losses, batch

    def plot_losses(self, log: bool = False, show: bool = False) -> None:
        """Plot training and validation losses."""
        plotting.plot_losses(
            self.losses,
            name="losses" + ("_log" if log else ""),
            plot_dir=self.save_dir,
            log=log,
            show=show,
        )

    def save_losses_csv(self, name: str = "losses.csv") -> Path:
        """
        Save the losses dictionary to a CSV file.

        Args:
            name: Filename for the CSV

        Returns:
            Path to saved CSV file
        """
        # Prepare data for DataFrame
        data = {}
        for key, values in self.losses.items():
            if key == "otp_alpha":
                # otp_alpha is stored as list of [epoch, value] pairs
                if values:
                    data["otp_alpha_epoch"] = [v[0] for v in values]
                    data["otp_alpha_value"] = [v[1] for v in values]
            else:
                data[key] = values

        # Create DataFrame (handles different-length lists with NaN padding)
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))

        save_path = self.save_dir / name
        df.to_csv(save_path, index=False)
        self.logger.info(f"Losses saved to {save_path}")
        return save_path

    def post_training(self, show: bool = False) -> Path:
        """
        Run post-training tasks: plot losses and save final model.

        Returns:
            Path to saved model checkpoint
        """
        self.plot_losses(show=show)
        self.save_losses_csv()
        save_path = self.save_checkpoint("model.pt")
        self.logger.info(f"Model saved to {save_path}")
        return save_path

    def _get_item(self, loss: Any) -> float:
        """Safely extract scalar value from loss tensor."""
        try:
            return loss.item()
        except (ValueError, AttributeError):
            return float(loss)

    def _process_batch(self, batch: list[Tensor]) -> Tensor:
        """Convert list of samples per timepoint into tensor of shape (batch_size, num_timepoints, dim)."""
        return torch.stack(batch).transpose(0, 1)
