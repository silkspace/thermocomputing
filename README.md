# The Criticality Engine

**Learning in thermodynamic systems via the Onsager-Machlup action.**

---

## The Problem

Modern ML simulates physics on digital hardware. Neural networks, diffusion models, and energy-based models are discretizations of stochastic differential equations—yet we run them on GPUs using floating-point arithmetic, burning energy to *pretend* to be analog.

Thermodynamic computers—built from fluctuating devices like CMOS p-bits—naturally undergo the stochastic dynamics that digital systems struggle to simulate. The problem is training: no one has figured out how to learn parameters on-chip.

## The Breakthrough

Learning in thermodynamic systems is **constraint satisfaction, not optimization**.

The Onsager-Machlup action measures how well parameters explain observed trajectories. When parameters are correct, observed dynamics become maximally likely—not because we minimized a loss, but because we satisfied a physical constraint.

**Gradient formulas fall out analytically:**

```
∂L/∂b_i  = residual_i / (2kT)
∂L/∂J_ij = -(residual_i · x_j + residual_j · x_i) / (2kT)
```

Local measurements → local updates. No backpropagation. Hardware-friendly.

## The Key Result

**78.8% accuracy on data where template matching completely fails.**

We use within-class centering to remove all first-order (mean) information:

```
Nearest Centroid (raw data):     81.9%
Nearest Centroid (centered):      9.4%  ← random chance, means removed
Coupling Learning (centered):    78.8%  ← genuine second-order structure!
```

This proves the method learns **correlations between pixels**, not just templates.

## How It Works

### Thermal Pixel Exchanges

We observe mass-conserving dynamics where pixels exchange intensity (Kawasaki-like):

```
x_i → x_i - δ,   x_j → x_j + δ
```

Correlated pixels (both bright or both dark) produce different exchange statistics than uncorrelated pixels. The coupling gradient extracts this structure.

### The Full Potential

```
V_c(x) = J₂||x||² + J₄||x||⁴ + b_c·x + ½x^T J_c x
         ─────────────────────  ─────   ──────────
         φ⁴ bistability         bias    couplings
                               (1st order) (2nd order)
```

- **Biases** encode class templates (where attractors are)
- **Couplings** encode correlations (how pixels vary together)

## Quick Start

```bash
uv sync

# Run coupling learning (main experiment)
uv run python learn_couplings.py

# Generate digits with learned model
uv run python learn_couplings.py --reconstruct
```

## Repository Structure

```
├── learn_couplings.py          # Main implementation
├── utils.py                    # Data loading
├── morisse-whitelam.tex/pdf    # Full paper
├── paper_figures/              # Generated figures
└── archive/                    # Old experiments
```

## The Paper

Full theory in `morisse-whitelam.pdf`:

- Onsager-Machlup action as trajectory likelihood
- Local gradient estimators for b_i and J_ij
- Hardware architecture for thermodynamic computers
- Learning as constraint satisfaction

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
