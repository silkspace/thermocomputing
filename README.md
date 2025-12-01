# The Criticality Engine

**Trajectory-Based Parameter Estimation for Thermodynamic Neural Computers**

## Key Result

**76% MNIST test accuracy in ONE EPOCH** using analytical gradient formulas from the Onsager-Machlup action. No neural networks, no backpropagation, no autograd.

```
BEFORE TRAINING: Acc=10.0% (random baseline)
After 1 epoch:   Acc=76.4% (on holdout test set)
```

## Quick Start

```bash
# Run the main experiment
uv run python trajectory_estimator.py

# Generate denoising demo
uv run python denoise_with_learned_biases.py

# (Optional) Compare with PyTorch/autograd
uv run python pytorch_comparison.py
```

## Core Files

| File | Description |
|------|-------------|
| `trajectory_estimator.py` | **Main implementation** - Analytical gradient estimator |
| `denoise_with_learned_biases.py` | Denoising demo using learned biases |
| `pytorch_comparison.py` | PyTorch comparison (shows autograd is slower) |
| `TRAJECTORY_ESTIMATION.md` | Detailed explanation of the method |
| `morisse-whitelam.tex` | The paper manuscript |
| `paper_figures/` | All figures for the paper |
| `archive/` | Old experiments (score matching, etc.) |

## The Method

### Physics Ansatz

We use a **phi-4 + bias potential** (not a neural network):

```
V_c(x) = J_2 ||x||^2 + J_4 ||x||^4 + b_c . x
```

Where:
- `J_2 = -1.0`: Creates bistability (double-well per pixel)
- `J_4 = 0.5`: Bounds the potential
- `b_c`: Learnable bias for class c (196-dimensional for 14x14 images)

Classification: `c* = argmin_c V_c(x)`

### Analytical Gradient Estimators

From the Onsager-Machlup action (paper Eq 12-13):

```
residual = observed_displacement - predicted_displacement
         = dx + mu * grad_V * dt

gradient for b = -residual / (2kT)
```

**This is NOT autograd.** We compute gradients analytically from observed trajectory residuals.

### Conservative Pixel Diffusion

The key insight: diffusion must be **data-dependent** to learn class structure.

```python
# Each pixel exchanges intensity with neighbors (Laplacian diffusion)
# Different digits have different spatial structures -> different diffusion patterns
# Gradients encode these class-specific patterns
```

## Results

### Holdout Test Accuracy

| Digit | Accuracy |
|-------|----------|
| 0 | 98.0% |
| 1 | 83.2% |
| 2 | 54.5% |
| 3 | 84.3% |
| 4 | 82.6% |
| 5 | 44.7% |
| 6 | 74.5% |
| 7 | 87.0% |
| 8 | 76.6% |
| 9 | 70.0% |
| **Overall** | **76.4%** |

### Why It Works So Fast

1. **Conservative diffusion is data-dependent**: Different digits diffuse differently
2. **Gradient estimator extracts templates**: Mean residual per class = class template
3. **One update is enough**: Biases become template-matching discriminators
4. **No optimization needed**: This is constraint satisfaction, not loss minimization

## Comparison with Standard ML

| Approach | Accuracy | Training Time | Autograd? |
|----------|----------|---------------|-----------|
| Trajectory Estimation | 76% | ~30s | No |
| PyTorch (velocity matching) | ~75% | ~90s | Yes |
| Linear classifier baseline | ~85% | ~1s | Yes |

Our method achieves competitive accuracy with analytical gradients only.

## What's NOT in This Repo

The `archive/` folder contains old experiments that used:
- Score matching (different objective)
- Neural network training (different paradigm)
- Unconditional generation (doesn't work for multi-modal)

**These approaches gave 37% accuracy or less.** The trajectory estimation method in the main repo gives 76%.

## Citation

```bibtex
@article{morisse2024criticality,
  title={The Criticality Engine: Online Learning in Thermodynamic Neural Computers},
  author={Morisse, Alexander and Whitelam, Stephen},
  year={2024}
}
```
