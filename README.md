<!-- <p align="center">
    <img src="media/method_overview.png" alt="Method Overview"/>
</p> -->

<p align="center">
<b>OTP-FM: Multimarginal flow matching (FM) with optimal transport potentials (OTP) (ICML 2026)</b>
</p>

---

<p align="center">
  <a href="#overview">Paper</a> •
  <a href="#overview">Overview</a> •
  <a href="#why-otp-fm">Why OTP-FM?</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#tutorials">Tutorials</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#citation">Citation</a>
</p>

---

# OTP-FM: Multimarginal flow matching (FM) with optimal transport potentials (OTP)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/Bexorg-Inc/OTP-FM/actions/workflows/ci.yml/badge.svg)](https://github.com/Bexorg-Inc/OTP-FM/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Bexorg-Inc/OTP-FM/badges/coverage-badge.json)](https://github.com/Bexorg-Inc/OTP-FM/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

A PyTorch library for training flow matching models with intermediate marginal constraints enforced using "optimal transport potentials". Includes code for reproducing [Kansal et. al., ICML 2026]().

## Overview

OTP-FM extends vanilla conditional flow matching (CFM) between endpoint marginals to incorporate intermediate marginal constraints as well.
We do so by modifying the **dynamic optimal transport** problem to incorporate **potential energy** terms corresponding to these intermediate marginals and updating the CFM targets based on the resulting dynamics.

## Why OTP-FM?
 - Flexibility in the choice of potentials and temporal dynamics
 - Efficient training for a variety of potentials; in particular, linear time training with the $\mathcal W_2^\infty$ (`W2Inf`) potential
 - Stable training using the OTPFM curriculum
 - SOTA results in multimarginal inference tasks

Check out [Quick Start](#quick-start) and [Tutorials](#tutorials) to see it in action.

## Installation

### For Users (pip)

```bash
# Core package
pip install otpfm

# With W2Potential support (requires POT library)
pip install otpfm[w2]
```

### To run experiments or develop (pixi)

[Pixi](https://pixi.sh) is a fast conda-like package manager. Install it first:

```bash
curl -sSf https://pixi.sh/install.sh | bash
```

Then set up the environment:

```bash
git clone https://github.com/Bexorg-Inc/OTP-FM.git
cd otpfm
pixi install
pixi shell
```

Both the `otpfm` and `experiments` packages will be installed.

## Quick Start

```python
import torch
from collections import OrderedDict
from otpfm import OTPFM, Curriculum
from otpfm.potentials import W2InfPotential

# Training data: samples from each marginal (batch_size, num_marginals, dim)
# For K=2 intermediate times: [source, t=0.33, t=0.67, target]
xs = torch.randn(64, 4, 2)

# Define K = 2 intermediate marginal potentials
tks = [0.33, 0.67]  # Intermediate time points
potentials = OrderedDict({
    tks[0]: W2InfPotential(tk=tks[0], strength=100.0, lambda_type='gaussian', width=0.2),
    tks[1]: W2InfPotential(tk=tks[1], strength=100.0, lambda_type='gaussian', width=0.2),
})

# Create model
model = OTPFM(
    d=2,                          # Data dimension
    tks=tks,                      # Intermediate time points
    potentials=potentials,        # OT potentials
    flownet_args={
        'hidden_dim': 128,
        'num_hidden_layers': 2,
    }
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 100
iterations_per_epoch = 50

# This controls transition from vanilla flow matching (alpha=0) to full OTP-FM (alpha=1)
otp_alpha_schedule = Curriculum(total_iterations=n_epochs * iterations_per_epoch)  # Sigmoid schedule by default

iterations = 0
for epoch in range(n_epochs):
    for batch_idx in range(steps_per_epoch):
        model.train()

        otp_alpha = otp_alpha_schedule(iterations)

        # Forward pass
        loss = model.forward_with_loss(xs, otp_alpha=otp_alpha)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update EMA model
        model.update_ema()
        iterations += 1

# Sample trajectories
model.eval()
x0 = torch.randn(100, 2)  # Initial samples
with torch.no_grad():
    trajectories, times = model.sample(x0, n_steps=10)
```

## Tutorials

- [**01_quickstart_gaussians.ipynb**](notebooks/01_quickstart_gaussians.ipynb): Quickstart on synthetic gaussian data.
- [**02_singlecell_eb.ipynb**](notebooks/02_singlecell_eb.ipynb): Embryoid body scRNA-seq data.
- [**03_gulf_of_mexico.ipynb**](notebooks/03_gulf_of_mexico.ipynb): Modeling ocean currents in the Gulf of Mexico.
- [**04_beijing_airquality.ipynb**](notebooks/04_beijing_airquality.ipynb): Beijing air quality data.
- [**05_exact_gaussian_solutions.ipynb**](notebooks/05_exact_gaussian_solutions.ipynb): Exact solutions for dynamic OT with potentials for Gaussian marginals.

## Customization

### Potential Types

OTP-FM supports multiple potentials based on different statistical distances:

```python
from otpfm.potentials import (
    W2InfPotential,          # Random coupling between samples; fastest and default recommendation
    W2Potential,             # Exact Wasserstein-2 (requires pot); usually better to use W2InfPotential with pre-computed OT couplings
    MMDRBFPotential,         # MMD with RBF kernel
    KLPotential,             # KL divergence with score estimation
)
```

### Spatiotemporal dynamics

Spatial and temporal dynamics can be tuned by changing the strengths, widths, and shapes of the potentials, e.g.:

```python
tks = [0.1, 0.5, 0.7]  # Intermediate time points
potentials = OrderedDict({
    tks[0]: W2InfPotential(tk=tks[0], strength=500.0, lambda_type='box', width=0.1),
    tks[1]: W2InfPotential(tk=tks[1], strength=400.0, lambda_type='triangle', width=0.2),
    tks[2]: W2InfPotential(tk=tks[2], strength=100.0, lambda_type='gaussian', width=0.05),
})
```

### Custom Velocity Networks

You can provide your own velocity network:

```python
import torch.nn as nn

class MyVelocityNet(nn.Module):
    def forward(self, x, t1, dt):
        """
        Args:
            x: (batch, d) positions
            t1: (batch,) start times
            dt: (batch,) time intervals
        Returns:
            v: (batch, d) velocities
        """
        # Your implementation
        pass

model = OTPFM(
    d=2,
    tks=[0.5],
    potentials=potentials,
    flownet=MyVelocityNet()  # Custom network
)
```

## Citation

If you use this code in your research, please cite:

```
Coming soon.
```

## Reproducing Experiments

For reproducing the experiments from the ICML paper, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
