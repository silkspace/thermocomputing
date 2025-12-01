# The Criticality Engine

**A thermodynamic approach to machine learning that doesn't need backpropagation.**

---

## The Problem

Modern machine learning is computationally expensive because we *simulate* physics on digital hardware. Neural networks, diffusion models, and energy-based models are all discretizations of stochastic differential equations—yet we run them on GPUs using floating-point arithmetic, burning energy to *pretend* to be analog.

What if we stopped pretending?

## The Idea

Thermodynamic computers—built from fluctuating electronic devices like CMOS p-bits, nanomagnetic oscillators, or analog relaxation circuits—naturally undergo the stochastic dynamics that digital systems struggle to simulate. These devices don't compute Langevin equations; they *are* Langevin equations.

The problem is training. Existing thermodynamic hardware can do inference (sampling, optimization), but no one has figured out how to estimate gradients and update parameters on-chip. Without training, you're stuck with fixed models.

## The Breakthrough

We discovered that learning in thermodynamic systems is **constraint satisfaction, not optimization**.

The [Onsager-Machlup action](https://en.wikipedia.org/wiki/Onsager%E2%80%93Machlup_function) measures how well model parameters explain observed trajectories. When parameters are correct, observed dynamics become maximally likely—not because we minimized a loss, but because we satisfied a physical constraint. Temperature acts as a Lagrangian multiplier. Forward-backward symmetry provides additional constraints.

This explains why trajectory-based learning is fast: projecting onto a constraint surface is geometrically simpler than searching a high-dimensional loss landscape.

**The gradient formulas fall out analytically. No autograd. No backpropagation.**

```
gradient = (observed_displacement - predicted_displacement) / (2kT)
```

That's it. Local measurements, local updates. Hardware-friendly.

## The Result

**76% MNIST classification accuracy on a holdout test set—in a single epoch.**

```
BEFORE TRAINING: Acc=10.0% (random guess)
After 1 epoch:   Acc=76.4% (learned from trajectory statistics)
```

For comparison, conventional neural networks typically require thousands of epochs to converge. We achieve competitive accuracy with:
- No neural network
- No backpropagation
- No autograd
- Just physics

## Quick Start

```bash
# Install dependencies
uv sync

# Run the main experiment (takes ~30 seconds)
uv run python trajectory_estimator.py

# See denoising with learned attractors
uv run python denoise_with_learned_biases.py

# Compare against standard ML baselines
uv run python pytorch_comparison.py
```

## How It Works

### The Physics

We use a φ⁴ + bias potential—a bistable energy landscape commonly found in physical systems:

```
V_c(x) = J₂||x||² + J₄||x||⁴ + b_c·x
```

Each class c has a learnable bias vector b_c that shapes its attractor basin. Classification is energy minimization: assign x to the class whose potential well it falls into.

### The Learning Rule

From the Onsager-Machlup action, we derive analytical gradients:

```python
# Observe trajectory under diffusion
dx_observed = x_next - x

# Predict what the model expects
dx_predicted = -μ * ∇V * dt

# The gradient is just the residual
residual = dx_observed - dx_predicted
grad_b = -residual / (2kT)
```

No computational graph. No chain rule. The physics gives us the gradient directly.

### Why One Epoch Works

The diffusion process is **data-dependent**: different digits have different spatial structures, so they diffuse differently. When we average trajectory residuals over each class, we extract a *template* of that class. One pass through the data is enough to learn discriminative features.

## Results

| Digit | Accuracy | | Digit | Accuracy |
|-------|----------|--|-------|----------|
| 0 | 98.0% | | 5 | 44.7% |
| 1 | 83.2% | | 6 | 74.5% |
| 2 | 54.5% | | 7 | 87.0% |
| 3 | 84.3% | | 8 | 76.6% |
| 4 | 82.6% | | 9 | 70.0% |

**Overall: 76.4%** on holdout test set (1000 samples never seen during training)

### Comparison

| Method | Test Accuracy | Training Time | Backprop? |
|--------|---------------|---------------|-----------|
| **Trajectory Estimation** | 76% | 30s | No |
| PyTorch (same objective) | 76% | 90s | No* |
| Logistic Regression | 90% | 0.2s | Yes |

*PyTorch version uses our analytical gradients, not autograd.

We don't beat logistic regression on accuracy—but we don't need gradients to flow through a computational graph. This is the point: these learning rules can run on hardware where backpropagation is physically impossible.

## Repository Structure

```
├── trajectory_estimator.py     # Main implementation
├── denoise_with_learned_biases.py  # Denoising demo
├── pytorch_comparison.py       # Baseline comparisons
├── utils.py                    # Shared data loading
├── morisse-whitelam.tex        # Full paper
└── paper_figures/              # Generated figures
```

## The Paper

The full theoretical framework, hardware blueprint, and experimental validation are in `morisse-whitelam.pdf`. Key sections:

- **Onsager-Machlup action** as trajectory likelihood
- **Local gradient estimators** for biases and couplings
- **Hardware architecture** with measurement plane, statistics layer, and reconfiguration fabric
- **CMOS implementation** using p-bits at threshold

## Citation

```bibtex
@article{morisse2025criticality,
  title={The Criticality Engine: Online Learning in Thermodynamic Neural Computers},
  author={Morisse, Alexander and Whitelam, Stephen and Tamblyn, Isaac},
  year={2025}
}
```

## License

MIT

---

*What if the next generation of AI runs on thermal noise instead of fighting it?*
