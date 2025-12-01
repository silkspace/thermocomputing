# The Criticality Engine

**A thermodynamic approach to machine learning using the Onsager-Machlup action.**

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run coupling learning (the main experiment)
uv run python learn_couplings.py

# Run reconstruction demo with saved J matrices
uv run python learn_couplings.py --reconstruct
```

## The Key Result

**78.8% MNIST classification accuracy on within-class centered data** where nearest-centroid achieves only 9.4% (random chance).

This proves we learn **genuine second-order (correlational) structure**, not just templates.

```
Nearest Centroid (raw data):     81.9%
Nearest Centroid (centered):      9.4%  ← means removed, centroids fail
J Learning (centered):           78.8%  ← REAL second-order learning!
```

## How It Works

### Thermal Pixel Exchanges (Kawasaki Dynamics)

We observe **mass-conserving** stochastic dynamics where pixels exchange intensity:

```
x_i → x_i - δ,  x_j → x_j + δ
```

This conserves total mass and creates data-dependent diffusion: correlated pixels (both bright or both dark) produce different exchange statistics than uncorrelated pixels.

### The Learning Rule

From the Onsager-Machlup action, we derive closed-form gradients:

```python
# Coupling gradient from exchange observation
∂L/∂J_ij ∝ -(residual_i · x_j + residual_j · x_i) / (2kT)
```

No backpropagation. Local measurements → local updates.

### The Potential

The full thermodynamic potential combines:

```
V_c(x) = J₂||x||² + J₄||x||⁴ + b_c·x + ½x^T J_c x
         ────────────────────   ─────   ──────────
         φ⁴ bistability         bias    couplings
                               (template) (correlations)
```

- **Biases** encode first-order structure (class templates = class means)
- **Couplings** encode second-order structure (correlations between pixels)

## Repository Structure

```
├── learn_couplings.py          # Main implementation (thermal exchanges)
├── utils.py                    # Data loading utilities
├── morisse-whitelam.tex        # Full paper
├── morisse-whitelam.pdf        # Compiled paper
├── paper_figures/              # Generated figures
│   ├── thermalization_demo.png         # Thermal exchange dynamics
│   ├── exchange_learned_J_spatial.png  # Learned coupling patterns
│   ├── exchange_learning_curves.png    # Training curves
│   └── generated_digits.png            # Full model generation
└── archive/                    # Old experiments (for reference)
```

## Key Figures

| Figure | Description |
|--------|-------------|
| `thermalization_demo.png` | Mass-conserving thermal exchanges |
| `exchange_learned_J_spatial.png` | Learned J matrices (local correlation patterns) |
| `exchange_learning_curves.png` | 78.8% accuracy on centered data |
| `generated_digits.png` | Digit generation with full model |

## The Paper

Full theoretical framework in `morisse-whitelam.pdf`:

- **Onsager-Machlup action** as trajectory likelihood
- **Local gradient estimators** for biases b_i and couplings J_ij
- **Hardware architecture** for thermodynamic computers
- **Learning as constraint satisfaction** (not optimization)

## Citation

```bibtex
@article{morisse2025criticality,
  title={The Criticality Engine: Online Learning in Thermodynamic Neural Computers},
  author={Morisse, Alexander and Whitelam, Stephen and Tamblyn, Isaac},
  year={2025}
}
```

---

*Learning from thermal noise, not fighting it.*
