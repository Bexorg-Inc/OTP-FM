# OTP-FM: Multimarginal flow matching (FM) with optimal transport potentials (OTP)

A PyTorch library for training flow matching models with intermediate marginal constraints enforced using "optimal transport potentials".

- [OTP-FM: Multimarginal flow matching (FM) with optimal transport potentials (OTP)](#otp-fm-multimarginal-flow-matching-fm-with-optimal-transport-potentials-otp)
  - [Overview](#overview)
  - [Why OTP-FM?](#why-otp-fm)
  - [Installation](#installation)
    - [For Users (pip)](#for-users-pip)
    - [For Developers (pixi)](#for-developers-pixi)
  - [Quick Start](#quick-start)
  - [Tutorials](#tutorials)
  - [Potential Types](#potential-types)
  - [Custom Velocity Networks](#custom-velocity-networks)
  - [Citation](#citation)
  - [License](#license)
  - [Reproducing Experiments](#reproducing-experiments)


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

# With ODE-based sampling (requires torchdiffeq)
pip install otpfm[ode]

# All optional dependencies
pip install otpfm[all]
```

### For Developers (pixi)

[Pixi](https://pixi.sh) is a fast conda-like package manager. Install it first:

```bash
curl -sSf https://pixi.sh/install.sh | bash
```

Then set up the development environment:

```bash
pixi install
pixi shell
```

## Quick Start

```python
import torch
from collections import OrderedDict
from otpfm import OTPFM, Curriculum
from otpfm.potentials import W2InfPotential

# Training data: samples from each marginal (batch_size, num_marginals, dim)
# For K=2 intermediate times: [source, t=0.33, t=0.67, target]
xs = torch.randn(64, 4, 2)

# Define K = 2 intermediate marginal constraints
tks = [0.33, 0.67]  # Intermediate time points
potentials = OrderedDict({
    tks[0]: W2InfPotential(tk=tks[0], strength=100.0, lambda_fn_type='gaussian', width=0.2),
    tks[1]: W2InfPotential(tk=tks[1], strength=100.0, lambda_fn_type='gaussian', width=0.2),
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
# This controls transition from vanilla flow matching (alpha=0) to full OTP-FM (alpha=1)
otp_alpha_schedule = Curriculum(total_steps=n_epochs)  # Sigmoid schedule by default

for epoch in range(n_epochs):
    model.train()
    otp_alpha = alpha_schedule(epoch)
    
    # Forward pass
    loss = model.forward_with_losses(xs, otp_alpha=otp_alpha)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Update EMA model (used for stable sampling)
    model.update_ema()

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


## Potential Types

OTP-FM supports potentials based on different statistical distances:

```python
from otpfm.potentials import (
    W2InfPotential,          # Random coupling between samples, fastest and default recommendation for most applications
    W2Potential,             # Exact Wasserstein-2 (requires pot)
    MMDRBFPotential,         # MMD with RBF kernel
    KLPotential,             # KL divergence with score estimation
)
```

## Custom Velocity Networks

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
TODO
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Reproducing Experiments

For reproducing the experiments from the ICML paper, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).