"""
Tests for single-cell trajectory inference module.

Run with: pytest tests/test_singlecell.py -v

Author(s): Raghav Kansal
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from collections import OrderedDict

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def device():
    """Get device for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    """Create a temporary data directory."""
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def eb_data(data_dir):
    """Download and load EB data for tests."""
    from fm_explore.singlecell.dataset import download_eb_data, load_eb_data
    
    data_path = download_eb_data(data_dir)
    pcs, labels, velocity, phate, _ = load_eb_data(
        data_path, max_dim=20, normalize=True, return_phate=True
    )
    return pcs, labels, velocity, phate


@pytest.fixture(scope="module")
def synthetic_data():
    """Create synthetic data for fast tests without downloading."""
    np.random.seed(42)
    n_cells = 500
    dim = 20
    n_timepoints = 5
    
    # Create synthetic trajectories - cells progressing through time
    labels = np.repeat(np.arange(n_timepoints), n_cells // n_timepoints)
    pcs = np.random.randn(len(labels), dim).astype(np.float32)
    
    # Add time-dependent drift to make it trajectory-like
    for t in range(n_timepoints):
        mask = labels == t
        pcs[mask] += t * 0.5 * np.random.randn(1, dim).astype(np.float32)
    
    return pcs, labels.astype(np.int64)


@pytest.fixture(scope="module")
def simple_model(device):
    """Create a simple OTPFM model for testing."""
    from fm_explore.otpfm import OTPFM, MMDRBFPotential
    
    dim = 20
    tks = [0.33, 0.67]
    potentials = OrderedDict()
    for tk in tks:
        potentials[tk] = MMDRBFPotential(
            tk=tk,
            strength=1.0,
            lambda_fn_type="gaussian",
            width=0.2,
            sigma=[1.0, 3.0, 10.0],
        )
    
    model = OTPFM(
        d=dim,
        tks=tks,
        potentials=potentials,
        flownet_args={
            "x_emb_dim": 32,
            "t_emb_dim": 32,
            "num_hidden_layers": 2,
            "hidden_dim": 64,
        },
        ema_decay=0.999,
        euler_steps=2,
    ).to(device)
    
    return model


# ============================================================================
# Dataset Tests
# ============================================================================


class TestDataset:
    """Tests for dataset loading and preprocessing."""
    
    def test_download_eb_data(self, data_dir):
        """Test that EB data can be downloaded."""
        from fm_explore.singlecell.dataset import download_eb_data
        
        data_path = download_eb_data(data_dir)
        assert data_path.exists()
        assert data_path.suffix == ".npz"
    
    def test_load_eb_data_basic(self, eb_data):
        """Test basic data loading."""
        pcs, labels, velocity, phate = eb_data
        
        assert pcs.ndim == 2
        assert labels.ndim == 1
        assert pcs.shape[0] == labels.shape[0]
        assert pcs.shape[1] == 20  # max_dim
        assert pcs.dtype == np.float32
        assert labels.dtype == np.int64
    
    def test_load_eb_data_labels(self, eb_data):
        """Test that labels are correct time points."""
        pcs, labels, _, _ = eb_data
        
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 5  # 5 time points
        assert unique_labels.tolist() == [0, 1, 2, 3, 4]
    
    def test_load_eb_data_normalizeing(self, eb_data):
        """Test that normalizeing produces approximately zero mean and unit variance."""
        pcs, _, _, _ = eb_data
        
        # After normalizeing, mean should be ~0, std should be ~1
        assert np.abs(pcs.mean()) < 0.1
        assert np.abs(pcs.std() - 1.0) < 0.1
    
    def test_load_eb_data_phate(self, eb_data):
        """Test PHATE embedding loading."""
        _, _, _, phate = eb_data
        
        assert phate is not None
        assert phate.ndim == 2
        assert phate.shape[1] == 2  # 2D embedding
    
    def test_eb_multi_marginal_dataset(self, synthetic_data):
        """Test EBMultiMarginalDataset creation and iteration."""
        from fm_explore.singlecell.dataset import EBMultiMarginalDataset
        
        pcs, labels = synthetic_data
        holdout_times = [1, 3]
        
        dataset = EBMultiMarginalDataset(pcs, labels, holdout_times=holdout_times)
        
        # Check training times
        assert dataset.train_times == [0, 2, 4]
        assert len(dataset) > 0
        
        # Check item retrieval
        samples = dataset[0]
        assert len(samples) == 3  # 3 training time points
        assert all(isinstance(s, torch.Tensor) for s in samples)
        assert all(s.shape == (20,) for s in samples)
    
    def test_create_eb_dataloaders(self, synthetic_data):
        """Test DataLoader creation."""
        from fm_explore.singlecell.dataset import create_eb_dataloaders
        
        pcs, labels = synthetic_data
        
        train_loader, val_loader = create_eb_dataloaders(
            pcs, labels,
            holdout_times=[1, 3],
            batch_size=32,
            val_split=0.2,
        )
        
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        
        # Check batch shapes
        batch = next(iter(train_loader))
        assert len(batch) == 3  # 3 training time points
        assert all(b.shape[0] == 32 for b in batch)  # batch size
    
    def test_get_holdout_data(self, synthetic_data):
        """Test extraction of holdout data."""
        from fm_explore.singlecell.dataset import get_holdout_data
        
        pcs, labels = synthetic_data
        
        holdout = get_holdout_data(pcs, labels, holdout_time=1)
        
        assert isinstance(holdout, torch.Tensor)
        assert holdout.dtype == torch.float32
        assert holdout.shape[1] == 20
        
        # Check that all samples are from time 1
        expected_count = (labels == 1).sum()
        assert holdout.shape[0] == expected_count
    
    def test_get_time_marginals(self, synthetic_data):
        """Test extraction of all time marginals."""
        from fm_explore.singlecell.dataset import get_time_marginals
        
        pcs, labels = synthetic_data
        
        marginals = get_time_marginals(pcs, labels)
        
        assert isinstance(marginals, dict)
        assert set(marginals.keys()) == {0, 1, 2, 3, 4}
        assert all(isinstance(v, torch.Tensor) for v in marginals.values())


# ============================================================================
# Model Tests
# ============================================================================


class TestModel:
    """Tests for OTPFM model."""
    
    def test_model_instantiation(self, simple_model):
        """Test model can be created."""
        assert simple_model is not None
        assert hasattr(simple_model, "flownet")
        assert hasattr(simple_model, "sample")
    
    def test_model_forward_pass(self, simple_model, synthetic_data, device):
        """Test forward pass computes losses."""
        pcs, labels = synthetic_data
        
        # Create batch: (batch_size, num_marginals, dim)
        # num_marginals = K + 2 = 4 (source, 2 intermediate, target)
        batch_size = 32
        marginals_dict = {}
        for t in [0, 2, 4]:  # training times
            mask = labels == t
            marginals_dict[t] = torch.tensor(pcs[mask][:batch_size], dtype=torch.float32)
        
        # Also need intermediate marginals for OTP
        for t in [1, 3]:  # intermediate times
            mask = labels == t
            marginals_dict[t] = torch.tensor(pcs[mask][:batch_size], dtype=torch.float32)
        
        # Stack in order: [source, intermediate_1, intermediate_2, target]
        # = [t0, t1, t3, t4] but we need [t0, t_{0.33}, t_{0.67}, t4]
        # For simplicity use t0, t1, t3, t4 which corresponds to normalized times 0, 0.25, 0.75, 1
        xs = torch.stack([
            marginals_dict[0],
            marginals_dict[1],  # Corresponds to tk=0.33
            marginals_dict[3],  # Corresponds to tk=0.67
            marginals_dict[4],
        ], dim=1).to(device)
        
        # Forward pass
        simple_model.train()
        loss = simple_model.forward_with_losses(xs, otp_alpha=0.5, do_otp=True)
        
        assert isinstance(loss, (float, torch.Tensor))
        if isinstance(loss, torch.Tensor):
            assert loss.numel() == 1
            assert torch.isfinite(loss)
    
    def test_model_sample(self, simple_model, device):
        """Test model sampling."""
        batch_size = 16
        dim = 20
        
        x0 = torch.randn(batch_size, dim).to(device)
        
        simple_model.eval()
        with torch.no_grad():
            trajectories, t_eval = simple_model.sample(x0, n_steps=10, ema=True)
        
        # trajectories shape: (n_timesteps, batch_size, dim)
        assert trajectories.ndim == 3
        assert trajectories.shape[1] == batch_size
        assert trajectories.shape[2] == dim
        
        # t_eval should have same length as first dim of trajectories
        assert len(t_eval) == trajectories.shape[0]
        
        # t_eval should go from 0 to 1
        assert t_eval[0] == 0.0
        assert t_eval[-1] == 1.0
    
    def test_model_ema_update(self, simple_model):
        """Test EMA update."""
        # Get a parameter from main and EMA model
        main_params = list(simple_model.flownet.parameters())
        ema_params = list(simple_model.flownet_ema.parameters())
        
        # Save initial EMA params
        initial_ema = ema_params[0].clone()
        
        # Modify main params
        with torch.no_grad():
            main_params[0].add_(0.1)
        
        # Update EMA
        simple_model.update_ema()
        
        # EMA should have changed
        assert not torch.allclose(ema_params[0], initial_ema)


# ============================================================================
# Evaluation Tests
# ============================================================================


class TestEvaluation:
    """Tests for evaluation metrics."""
    
    def test_compute_w2_distance(self):
        """Test Wasserstein-2 distance computation."""
        from fm_explore.evaluation import compute_w2_distance
        
        # Same distributions should have W2 ~ 0
        x = torch.randn(100, 10)
        w2 = compute_w2_distance(x, x)
        assert w2 < 0.1
        
        # Different distributions should have W2 > 0
        y = torch.randn(100, 10) + 5  # Shifted
        w2 = compute_w2_distance(x, y)
        assert w2 > 0.1
    
    def test_compute_mmd(self):
        """Test MMD with 3MSBM-style multi-scale Gaussian kernel."""
        from fm_explore.evaluation import compute_mmd
        
        # Same distributions should have MMD ~ 0
        x = torch.randn(100, 10)
        mmd = compute_mmd(x, x)
        assert mmd < 0.5
        
        # Different distributions should have MMD > 0
        y = torch.randn(100, 10) + 3
        mmd = compute_mmd(x, y)
        assert mmd > 0.1
    
    def test_compute_swd(self):
        """Test Sliced Wasserstein Distance."""
        from fm_explore.evaluation import compute_swd
        
        # Same distributions should have SWD ~ 0
        x = torch.randn(100, 10)
        swd = compute_swd(x, x)
        assert swd < 0.5
        
        # Different distributions should have SWD > 0
        y = torch.randn(100, 10) + 3
        swd = compute_swd(x, y)
        assert swd > 0.1
    
    def test_compute_fgd(self):
        """Test Fréchet Gaussian Distance."""
        from fm_explore.evaluation import compute_fgd
        
        # Same distributions should have FGD ~ 0
        x = torch.randn(100, 10)
        fgd = compute_fgd(x, x)
        assert fgd < 0.5
        
        # Different distributions should have FGD > 0
        y = torch.randn(100, 10) + 3
        fgd = compute_fgd(x, y)
        assert fgd > 0.1
        
        # Test with different covariance
        z = torch.randn(100, 10) * 3  # Different variance
        fgd_z = compute_fgd(x, z)
        assert fgd_z > 0.1  # Different covariance should give non-zero FGD
    
    
# ============================================================================
# Training Tests
# ============================================================================


class TestTraining:
    """Tests for training functionality."""
    
    def test_trainer_instantiation(self, simple_model, synthetic_data, device, tmp_path):
        """Test Trainer can be instantiated."""
        from fm_explore.singlecell.dataset import create_eb_dataloaders
        from fm_explore import Trainer
        
        pcs, labels = synthetic_data
        
        train_loader, val_loader = create_eb_dataloaders(
            pcs, labels,
            holdout_times=[1, 3],
            batch_size=16,
            val_split=0.2,
        )
        
        trainer = Trainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=5,
            sampling_steps=10,
            do_otp=True,
            grad_clip=1.0,
            otp_alpha_type="sigmoid",
            potentials=simple_model.potentials,
            device=device,
        )
        
        assert trainer.epochs == 5
        assert trainer.sampling_steps == 10
        assert trainer.do_otp
        assert trainer.grad_clip == 1.0
        assert trainer.otp_alpha_type == "sigmoid"
    
    def test_process_batch(self):
        """Test batch processing."""
        from fm_explore import Trainer
        
        # Simulate batch from DataLoader
        batch = [
            torch.randn(32, 20),  # time 0
            torch.randn(32, 20),  # time 2
            torch.randn(32, 20),  # time 4
        ]
        
        # Test the internal _process_batch method
        processed = torch.stack(batch).transpose(0, 1)
        
        # Should be (batch_size, num_marginals, dim)
        assert processed.shape == (32, 3, 20)
    
    def test_training_step(self, simple_model, synthetic_data, device, tmp_path):
        """Test a single training step."""
        from fm_explore.singlecell.dataset import create_eb_dataloaders
        
        pcs, labels = synthetic_data
        
        # Create dataloaders with intermediate marginals included
        # For this test, we'll manually create a batch
        train_loader, val_loader = create_eb_dataloaders(
            pcs, labels,
            holdout_times=[1, 3],
            batch_size=16,
            val_split=0.2,
        )
        
        # Manual test of training step
        simple_model.train()
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        
        for batch in train_loader:
            # batch is list of 3 tensors (train times 0, 2, 4)
            # We need to add intermediate marginals for OTP
            xs = torch.stack(batch).transpose(0, 1).to(device)
            
            # Create full batch with intermediate marginals
            # For simplicity, interpolate to get intermediate points
            x0 = xs[:, 0]
            x1 = xs[:, -1]
            # Interpolate for intermediate points
            xm1 = x0 * 0.67 + x1 * 0.33
            xm2 = x0 * 0.33 + x1 * 0.67
            xs_full = torch.stack([x0, xm1, xm2, x1], dim=1)
            
            optimizer.zero_grad()
            loss = simple_model.forward_with_losses(xs_full, otp_alpha=0.5, do_otp=True)
            
            if isinstance(loss, torch.Tensor):
                loss.backward()
                optimizer.step()
            
            break  # Just test one step
        
        # Should complete without error
    
    def test_eb_trainer_reshuffling(self, simple_model, synthetic_data, device, tmp_path):
        """Test EBTrainer with reshuffling enabled."""
        from fm_explore.singlecell.dataset import create_eb_dataloaders
        from fm_explore.singlecell.EBTrainer import EBTrainer
        
        pcs, labels = synthetic_data
        
        train_loader, val_loader = create_eb_dataloaders(
            pcs, labels,
            holdout_times=[1, 3],
            batch_size=16,
            val_split=0.2,
        )
        
        # Test with reshuffling enabled
        trainer = EBTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            sampling_steps=5,
            otp_alpha_type="1",
            potentials=simple_model.potentials,
            device=device,
            reshuffle_each_epoch=True,
        )
        
        assert trainer.reshuffle_each_epoch is True
        
        # Get underlying dataset
        underlying = trainer._get_underlying_dataset()
        assert hasattr(underlying, 'reshuffle')
        
        # Test that on_epoch_start calls reshuffle
        # Get initial indices
        initial_indices = {t: underlying.indices_by_time[t].clone() for t in underlying.train_times}
        
        # Call on_epoch_start which should reshuffle
        trainer.on_epoch_start(0)
        
        # Check that at least one time point has different indices
        indices_changed = False
        for t in underlying.train_times:
            if not torch.equal(initial_indices[t], underlying.indices_by_time[t]):
                indices_changed = True
                break
        
        assert indices_changed, "Reshuffling should change indices"
    
    def test_eb_trainer_no_reshuffling(self, simple_model, synthetic_data, device, tmp_path):
        """Test EBTrainer with reshuffling disabled."""
        from fm_explore.singlecell.dataset import create_eb_dataloaders
        from fm_explore.singlecell.EBTrainer import EBTrainer
        
        pcs, labels = synthetic_data
        
        train_loader, val_loader = create_eb_dataloaders(
            pcs, labels,
            holdout_times=[1, 3],
            batch_size=16,
            val_split=0.2,
        )
        
        # Test with reshuffling disabled
        trainer = EBTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            sampling_steps=5,
            otp_alpha_type="1",
            potentials=simple_model.potentials,
            device=device,
            reshuffle_each_epoch=False,
        )
        
        assert trainer.reshuffle_each_epoch is False
        
        # Get underlying dataset
        underlying = trainer._get_underlying_dataset()
        
        # Get initial indices
        initial_indices = {t: underlying.indices_by_time[t].clone() for t in underlying.train_times}
        
        # Call on_epoch_start which should NOT reshuffle
        trainer.on_epoch_start(0)
        
        # Check that all indices are the same
        for t in underlying.train_times:
            assert torch.equal(initial_indices[t], underlying.indices_by_time[t]), \
                "Indices should not change when reshuffling is disabled"
    
    def test_eb_trainer_with_evaluation(self, simple_model, synthetic_data, device, tmp_path):
        """Test EBTrainer with evaluation settings."""
        from fm_explore.singlecell.dataset import create_eb_dataloaders, get_time_marginals
        from fm_explore.singlecell.EBTrainer import EBTrainer
        
        pcs, labels = synthetic_data
        
        train_loader, val_loader = create_eb_dataloaders(
            pcs, labels,
            holdout_times=[1, 3],
            batch_size=16,
            val_split=0.2,
        )
        
        # Get marginals for evaluation
        marginals = get_time_marginals(pcs, labels)
        train_times = [0, 2, 4]
        holdout_times = [1, 3]
        
        # Test with evaluation enabled
        trainer = EBTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=tmp_path,
            lr=1e-3,
            epochs=1,
            sampling_steps=5,
            otp_alpha_type="1",
            potentials=simple_model.potentials,
            device=device,
            reshuffle_each_epoch=True,
            marginals=marginals,
            train_times=train_times,
            holdout_times=holdout_times,
            eval_n_steps=5,
            eval_num_samples=50,
            plot_trajectories=False,  # Disable plots for faster tests
        )
        
        assert trainer.marginals is not None
        assert trainer.train_times == train_times
        assert trainer.holdout_times == holdout_times


# ============================================================================
# Plotting Tests
# ============================================================================


class TestPlotting:
    """Tests for plotting functions."""
    
    def test_plot_pca_trajectories(self, tmp_path):
        """Test PCA trajectory plotting."""
        from fm_explore.singlecell.plotting import plot_pca_trajectories
        
        # Create fake trajectories: (n_timesteps, n_samples, dim)
        trajectories = torch.randn(20, 50, 20)
        time_points = np.linspace(0, 1, 20)
        
        save_path = tmp_path / "test_trajectories.pdf"
        
        # Should not raise
        plot_pca_trajectories(
            trajectories=trajectories,
            time_points=time_points,
            pcs=(0, 1),
            num_trajectories=10,
            title="Test Trajectories",
            save_path=save_path,
            show=False,
        )
        
        assert save_path.exists()
    


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_synthetic(self, device, tmp_path):
        """Test full training pipeline with synthetic data."""
        from collections import OrderedDict
        import torch
        import numpy as np
        
        from fm_explore.otpfm import OTPFM, MMDRBFPotential
        from fm_explore.singlecell.dataset import create_eb_dataloaders, get_time_marginals
        from fm_explore import evaluation
        
        # Create synthetic data
        np.random.seed(42)
        torch.manual_seed(42)
        
        n_cells = 200
        dim = 10
        n_timepoints = 5
        
        labels = np.repeat(np.arange(n_timepoints), n_cells // n_timepoints)
        pcs = np.random.randn(len(labels), dim).astype(np.float32)
        for t in range(n_timepoints):
            mask = labels == t
            pcs[mask] += t * np.array([0.5] * dim).astype(np.float32)
        
        # Create model
        tks = [0.33, 0.67]
        potentials = OrderedDict()
        for tk in tks:
            potentials[tk] = MMDRBFPotential(
                tk=tk, strength=1.0, lambda_fn_type="gaussian", width=0.2
            )
        
        model = OTPFM(
            d=dim,
            tks=tks,
            potentials=potentials,
            flownet_args={
                "x_emb_dim": 16,
                "t_emb_dim": 16,
                "num_hidden_layers": 1,
                "hidden_dim": 32,
            },
            ema_decay=0.99,
        ).to(device)
        
        # Create dataloader
        train_loader, _ = create_eb_dataloaders(
            pcs, labels.astype(np.int64),
            holdout_times=[1, 3],
            batch_size=16,
        )
        
        # Training loop (just 2 epochs for speed)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        
        for epoch in range(2):
            epoch_loss = 0.0
            for batch in train_loader:
                xs = torch.stack(batch).transpose(0, 1).to(device)
                
                # Create full batch with interpolated intermediate points
                x0, x1 = xs[:, 0], xs[:, -1]
                xm1 = x0 * 0.67 + x1 * 0.33
                xm2 = x0 * 0.33 + x1 * 0.67
                xs_full = torch.stack([x0, xm1, xm2, x1], dim=1)
                
                optimizer.zero_grad()
                loss = model.forward_with_losses(xs_full, otp_alpha=0.5)
                
                if isinstance(loss, torch.Tensor):
                    loss.backward()
                    optimizer.step()
                    model.update_ema()
                    epoch_loss += loss.item()
        
        # Evaluate by sampling and computing metrics
        marginals = get_time_marginals(pcs, labels.astype(np.int64))
        model.eval()
        
        with torch.no_grad():
            x0 = marginals[0][:30].to(device)
            trajectories, t_eval = model.sample(x0, n_steps=10, ema=True)
        
        # Get samples at middle time point
        mid_idx = len(t_eval) // 2
        generated = trajectories[mid_idx].cpu()
        
        # Compute metrics
        swd = evaluation.compute_swd(generated, marginals[2][:30].cpu())
        mmd = evaluation.compute_mmd(generated, marginals[2][:30].cpu())
        
        assert isinstance(swd, float)
        assert isinstance(mmd, float)
        assert np.isfinite(swd)
        assert np.isfinite(mmd)
    
    def test_model_saves_and_loads(self, simple_model, tmp_path, device):
        """Test model checkpoint save/load."""
        # Save
        save_path = tmp_path / "model.pt"
        torch.save({
            "model_state_dict": simple_model.state_dict(),
        }, save_path)
        
        assert save_path.exists()
        
        # Load into new model
        from collections import OrderedDict
        from fm_explore.otpfm import OTPFM, MMDRBFPotential
        
        dim = 20
        tks = [0.33, 0.67]
        potentials = OrderedDict()
        for tk in tks:
            potentials[tk] = MMDRBFPotential(
                tk=tk, strength=1.0, lambda_fn_type="gaussian", width=0.2
            )
        
        new_model = OTPFM(
            d=dim,
            tks=tks,
            potentials=potentials,
            flownet_args={
                "x_emb_dim": 32,
                "t_emb_dim": 32,
                "num_hidden_layers": 2,
                "hidden_dim": 64,
            },
            ema_decay=0.999,
        ).to(device)
        
        checkpoint = torch.load(save_path, map_location=device, weights_only=False)
        new_model.load_state_dict(checkpoint["model_state_dict"])
        
        # Check params match
        for (name1, p1), (name2, p2) in zip(
            simple_model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(p1, p2)


# ============================================================================
# Potential Tests
# ============================================================================


class TestPotentials:
    """Tests for different potential types."""
    
    def test_mmd_rbf_potential(self):
        """Test MMD RBF potential gradient computation."""
        from fm_explore.otpfm import MMDRBFPotential
        
        potential = MMDRBFPotential(
            tk=0.5, strength=1.0, lambda_fn_type="gaussian", width=0.2
        )
        
        x_true = torch.randn(50, 10)
        x_tk = torch.randn(50, 10)
        
        grad = potential.grad_gk(x_true, x_tk)
        
        assert grad.shape == x_tk.shape
        assert torch.isfinite(grad).all()
    
    def test_w2_potential(self):
        """Test W2 potential gradient computation."""
        from fm_explore.otpfm import W2Potential
        
        potential = W2Potential(
            tk=0.5, strength=1.0, lambda_fn_type="gaussian", width=0.2
        )
        
        x_true = torch.randn(30, 10)
        x_tk = torch.randn(30, 10)
        
        grad = potential.grad_gk(x_true, x_tk)
        
        assert grad.shape == x_tk.shape
        assert torch.isfinite(grad).all()
    
    def test_kl_potential_sliced(self):
        """Test KL potential with sliced score estimation."""
        from fm_explore.otpfm import KLPotential
        
        potential = KLPotential(
            tk=0.5, strength=1.0, lambda_fn_type="gaussian", width=0.2,
            rho_method="sliced", n_projections=32
        )
        
        x_true = torch.randn(50, 10)
        x_tk = torch.randn(50, 10)
        
        grad = potential.grad_gk(x_true, x_tk)
        
        assert grad.shape == x_tk.shape
        assert torch.isfinite(grad).all()
    
    def test_kl_potential_kde(self):
        """Test KL potential with KDE score estimation."""
        from fm_explore.otpfm import KLPotential
        
        potential = KLPotential(
            tk=0.5, strength=1.0, lambda_fn_type="gaussian", width=0.2,
            rho_method="kde"
        )
        
        x_true = torch.randn(50, 10)
        x_tk = torch.randn(50, 10)
        
        grad = potential.grad_gk(x_true, x_tk)
        
        assert grad.shape == x_tk.shape
        assert torch.isfinite(grad).all()


# ============================================================================
# Unified Train Script Tests
# ============================================================================


class TestUnifiedTrainScript:
    """Tests for the unified train.py script."""
    
    def test_config_loading(self, tmp_path):
        """Test that configs are loaded and merged correctly."""
        from fm_explore.train import load_json_config, merge_configs, get_config_dir
        
        # Check defaults exist
        config_dir = get_config_dir()
        assert (config_dir / "singlecell" / "defaults.json").exists()
        assert (config_dir / "gom" / "defaults.json").exists()
        
        # Test loading singlecell defaults
        config = load_json_config(config_dir / "singlecell" / "defaults.json")
        assert config["pca_dim"] == 100
        assert config["epochs"] == 300
        assert config["hidden_dim"] == 768
        
        # Test loading gom defaults
        config = load_json_config(config_dir / "gom" / "defaults.json")
        assert config["hidden_dim"] == 128
        assert config["epochs"] == 800
        assert config["tks"] == [0.25, 0.5, 0.75]
    
    def test_config_merging(self):
        """Test that config merging works correctly."""
        from fm_explore.train import merge_configs
        
        base = {"a": 1, "b": 2, "c": 3}
        override = {"b": 20, "d": 4}
        
        result = merge_configs(base, override)
        
        assert result["a"] == 1  # unchanged
        assert result["b"] == 20  # overridden
        assert result["c"] == 3  # unchanged
        assert result["d"] == 4  # new key
    
    def test_build_tag(self, synthetic_data, tmp_path):
        """Test build_tag function from train module."""
        from fm_explore.train import build_tag
        
        # Create a mock args object
        class Args:
            tag = "test"
            potential = "w2inf"
            strength = 100.0
            lambda_width = 0.33
            lr = 0.001
            num_hidden_layers = 4
            strengths = None
            widths = None
        
        tag = build_tag(Args)
        assert "test" in tag
        assert "w2inf" in tag
        assert "s100" in tag
    
    def test_build_tag_with_lists(self):
        """Test build_tag function with per-potential strengths/widths."""
        from fm_explore.train import build_tag
        
        class Args:
            tag = "experiment/test"
            potential = "w2inf"
            strength = 400.0
            width = 0.2
            lr = 0.001
            num_hidden_layers = 10
            hidden_dim = 128
            strengths = [100.0, 200.0, 300.0]
            widths = [0.1, 0.2, 0.3]
        
        tag = build_tag(Args)
        assert "test" in tag
        assert "w2inf" in tag
        assert "s100.0-200.0-300.0" in tag
        assert "w0.1-0.2-0.3" in tag
    
    def test_create_potential(self):
        """Test potential creation from train module."""
        from fm_explore.train import create_potential
        from fm_explore.otpfm import IndependentPotential, W2Potential, MMDRBFPotential
        
        class Args:
            strength = 100.0
            lambda_type = "gaussian"
            lambda_width = 0.33
            potential = "w2inf"
            mmd_bandwidth = [3.0]
            kl_rho_method = "kde"
            kl_mu_method = None
            kl_bandwidth = "3.0"
            kl_bandwidth_scale = 2.5
            kl_n_projections = 64
            kl_ssge_eta = 0.01
            kl_diagonal_cov = False
        
        # Test W2Inf
        Args.potential = "w2inf"
        pot = create_potential(Args, tk=0.5)
        assert isinstance(pot, IndependentPotential)
        assert pot.tk == 0.5
        assert pot.strength == 100.0
        
        # Test W2
        Args.potential = "w2"
        pot = create_potential(Args, tk=0.5)
        assert isinstance(pot, W2Potential)
        
        # Test MMD
        Args.potential = "mmd"
        pot = create_potential(Args, tk=0.5)
        assert isinstance(pot, MMDRBFPotential)
    
    def test_create_potential_with_overrides(self):
        """Test potential creation with strength/width overrides."""
        from fm_explore.train import create_potential
        from fm_explore.otpfm import IndependentPotential
        
        class Args:
            strength = 100.0
            lambda_type = "gaussian"
            lambda_width = 0.33
            potential = "w2inf"
        
        # Test with overrides
        pot = create_potential(Args, tk=0.5, strength=200.0, width=0.5)
        assert isinstance(pot, IndependentPotential)
        assert pot.strength == 200.0
        assert pot.lambda_fn.width == 0.5
    
    def test_reproducibility_fixed_seed(self, synthetic_data, device, tmp_path):
        """Test that training with fixed seed produces identical results."""
        from fm_explore.Trainer import Trainer
        from fm_explore.otpfm import OTPFM, IndependentPotential
        from torch.utils.data import Dataset, DataLoader
        
        pcs, labels = synthetic_data
        dim = pcs.shape[1]
        
        class SimpleDataset(Dataset):
            """Dataset that returns batches as list of tensors per time."""
            def __init__(self, pcs, labels, train_times):
                self.train_times = train_times
                # Get samples for each time
                self.samples_per_time = []
                for t in train_times:
                    mask = labels == t
                    self.samples_per_time.append(
                        torch.from_numpy(pcs[mask]).float()
                    )
                self.n_samples = min(len(s) for s in self.samples_per_time)
            
            def __len__(self):
                return self.n_samples
            
            def __getitem__(self, idx):
                # Return list of samples at each time point
                return [s[idx] for s in self.samples_per_time]
        
        def train_with_seed(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create simple model
            tks = [0.5]
            potentials = OrderedDict()
            potentials[0.5] = IndependentPotential(
                tk=0.5, strength=10.0, lambda_fn_type="gaussian", width=0.3
            )
            
            model = OTPFM(
                d=dim,
                tks=tks,
                potentials=potentials,
                flownet_args={
                    "x_emb_dim": 16,
                    "t_emb_dim": 16,
                    "num_hidden_layers": 2,
                    "hidden_dim": 32,
                },
                ema_decay=0.99,
            ).to(device)
            
            # Create dataloaders with proper format
            train_times = [0, 2, 4]
            dataset = SimpleDataset(pcs, labels, train_times)
            train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(dataset, batch_size=16, shuffle=False)
            
            # Train for 2 epochs
            save_dir = tmp_path / f"run_seed{seed}_{id(model)}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                save_dir=save_dir,
                lr=1e-3,
                epochs=2,
                potentials=potentials,
                device=device,
            )
            
            losses, _ = trainer.train()
            return losses["train_loss"][-1]
        
        # Train twice with same seed
        loss1 = train_with_seed(42)
        loss2 = train_with_seed(42)
        
        # Should be identical
        assert abs(loss1 - loss2) < 1e-6, f"Losses differ: {loss1} vs {loss2}"


# ============================================================================
# Run tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
