# Research & Development Plan: Hardware, Extensions, and IP

This document outlines research findings and action plans for four key areas:
1. DIY chip design for home experimentation
2. Chip simulators for model validation
3. Extending to stacked/deep architectures
4. Intellectual property considerations

---

## 1. DIY Chip Design: Can You Build This At Home?

### TL;DR
**Yes, with FPGAs.** You cannot fabricate custom CMOS p-bits at home, but you can implement functionally equivalent stochastic computing on affordable FPGA development boards.

### Hardware Options (Ranked by Accessibility)

#### Option A: FPGA-Based P-Bits (Recommended)
**Cost: $200-500 | Difficulty: Medium | Timeline: 2-4 weeks**

FPGAs can implement p-bits using Linear Feedback Shift Registers (LFSRs) + lookup tables. This approach:
- Requires ~1000-1200 transistor-equivalents per p-bit
- Achieves pseudo-random behavior with periods longer than experiment duration
- Is well-documented in academic literature

**Recommended Boards:**
| Board | Price | FPGA | Notes |
|-------|-------|------|-------|
| [Zybo Z7-20](https://digilent.com/shop/zybo-z7-zynq-7000-arm-fpga-soc-development-board/) | $299 | Zynq-7020 | Best balance of cost/capability |
| [ZedBoard](https://digilent.com/shop/zedboard-zynq-7000-arm-fpga-soc-development-board/) | $495 | Zynq-7020 | More peripherals, larger community |
| [Arty-Z7](https://digilent.com/shop/arty-z7-zynq-7000-soc-development-board/) | $199 | Zynq-7010 | Budget option, fewer resources |

**Key Resource:** [FPGA_SC Library](https://github.com/hinata9276/FPGA_SC) - Stochastic computing implementation for Xilinx FPGAs with MATLAB simulation and Vivado HLS code.

**What You Can Build:**
- 100-2000 interconnected p-bits
- Fully programmable coupling matrix J
- Real-time bias updates via ARM core
- USB interface for data collection

#### Option B: Discrete Analog Circuits
**Cost: $50-200 | Difficulty: High | Timeline: 4-8 weeks**

Build p-bits from discrete components using:
- Chaotic oscillators (tent-map circuits)
- Noise-injection + comparator circuits
- RC relaxation oscillators

**Challenges:**
- Calibration required for each p-bit
- Temperature sensitivity
- Limited scalability (practical limit: ~10-50 p-bits)

#### Option C: Mixed-Signal IC Prototyping
**Cost: $5,000+ | Difficulty: Expert | Timeline: 6-12 months**

Use shuttle runs (shared wafer fabrication):
- [Efabless/Google MPW](https://efabless.com/) - Free 130nm runs
- [Europractice](https://www.europractice-ic.com/) - Academic access
- [MOSIS](https://www.mosis.com/) - Commercial prototyping

**Recent Academic Work:**
- [Fully CMOS p-bit with bistable resistor](https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/adfm.202307935) (2024)
- [NbOx metal-insulator transition p-bit](https://www.nature.com/articles/s41467-023-43085-6) (2023)

### Recommended Path Forward

```
Week 1-2: Order Zybo Z7-20, install Vivado WebPACK
Week 3:   Implement single p-bit (LFSR + sigmoid LUT)
Week 4:   Scale to 10x10 p-bit array with programmable couplings
Week 5:   Port trajectory_estimator.py learning rule to ARM core
Week 6:   Run MNIST classification on FPGA, compare to simulation
```

---

## 2. Chip Simulators: Testing Models Before Silicon

### Available Simulators

#### thermox (Normal Computing) - Recommended
**[GitHub](https://github.com/normal-computing/thermox) | JAX | Open Source**

Exact Ornstein-Uhlenbeck process simulation. No discretization error.

```python
import thermox
import jax.numpy as jnp

# Define OU process: dx = -A(x-b)dt + sqrt(D)dW
samples = thermox.sample(ts, x0, A, b, D, key)
log_p = thermox.log_prob(x_trajectory, ts, A, b, D)  # For MLE
```

**Key Features:**
- O(d³ + Nd²) complexity for N samples
- Gradient-compatible via `jax.grad`
- Linear algebra primitives: `thermox.linalg.solve`, `inv`, `expm`

**Relevance to Our Work:**
Our φ⁴ + bias potential is nonlinear, but thermox's log_prob could be adapted for trajectory likelihood estimation. The JAX infrastructure is directly reusable.

#### thrml (Extropic)
**[GitHub](https://github.com/extropic-ai/thrml) | JAX | Open Source**

Probabilistic graphical model simulator designed for Extropic's TSU hardware.

```python
import thrml
# Build and sample from energy-based models
# Compute gradients for training
```

**Key Features:**
- Block Gibbs sampling on sparse graphs
- Energy-based model gradients
- Simulates Extropic Z1 chip architecture
- 831 GitHub stars, active development

**Relevance to Our Work:**
thrml's gradient computation for EBMs could inform our stacked-layer extension. Their Denoising Thermodynamic Model (DTM) is conceptually similar to our trajectory-based learning.

#### SPICE Simulation (For Analog Designs)
If building analog p-bits, use:
- **LTspice** (free) - General analog simulation
- **ngspice** (open source) - Command-line, scriptable
- MTJ SPICE models available from [academic repositories](https://www.researchgate.net/publication/308728667)

### Validation Strategy

```
1. Implement model in thermox/thrml (JAX)
2. Verify learning dynamics match trajectory_estimator.py
3. Export trained parameters
4. Run inference on FPGA hardware
5. Compare accuracy: simulation vs hardware
```

---

## 3. Stacked Layers: Extending to Deep Networks

### The Challenge
Our current model is a single-layer energy-based classifier:
```
V_c(x) = J₂||x||² + J₄||x||⁴ + b_c·x
```

To compete with deep networks, we need multi-layer architectures.

### Existing Approaches

#### Equilibrium Propagation (EP)
**[Scellier & Bengio, 2017](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full)**

Two-phase contrastive learning:
1. **Free phase:** Network relaxes to equilibrium
2. **Nudged phase:** Output nudged toward target, new equilibrium

**Gradient formula:**
```
∂L/∂θ = (1/β) * (∂E/∂θ|nudged - ∂E/∂θ|free)
```

**Pros:** Local, hardware-friendly, proven on deep networks
**Cons:** Requires two equilibration phases (slow)

**Recent:** [Quantum EP using Onsager reciprocity](https://www.nature.com/articles/s41467-025-61665-6) (2025) - Directly connects EP to our Onsager-Machlup framework!

#### Deep Boltzmann Machines on P-Bit Hardware
**[Nature Electronics, 2024](https://www.nature.com/articles/s41928-024-01182-4)**

Sparse Ising machines (FPGA p-bits) successfully trained multi-layer Boltzmann networks.

**Key insight:** Sparsity enables scalable hardware implementation.

#### Our Trajectory-Based Extension (Proposed)

**Hypothesis:** We can extend trajectory-based learning to stacked layers by:

1. **Per-layer diffusion:** Each layer diffuses independently
2. **Inter-layer coupling:** Layers coupled via Ising-like J matrix
3. **Trajectory residuals:** Compute residuals at each layer
4. **Local gradients:** Update each layer's parameters from local residuals

**Architecture:**
```
Layer 0 (input): x₀ = data
Layer 1 (hidden): x₁ relaxes under V₁(x₁, x₀)
Layer 2 (hidden): x₂ relaxes under V₂(x₂, x₁)
Layer K (output): x_K relaxes under V_K(x_K, x_{K-1})

Coupling energy: E_coupling = Σ_k x_k · W_k · x_{k+1}
Total energy: E = Σ_k V_k(x_k) + E_coupling
```

**Gradient derivation:**
From Onsager-Machlup action on coupled system:
```
∂L/∂W_k = (residual_k ⊗ x_{k+1} + x_k ⊗ residual_{k+1}) / (2kT)
```

### Implementation Plan

```python
# Proposed: stacked_trajectory_estimator.py

class StackedTrajectoryEstimator:
    def __init__(self, layer_dims: list[int], n_classes: int):
        self.layers = [
            TrajectoryEstimator(d_in, d_out)
            for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:])
        ]
        self.coupling_weights = [
            np.zeros((d_in, d_out))
            for d_in, d_out in zip(layer_dims[:-1], layer_dims[1:])
        ]

    def forward_diffusion(self, x, n_steps):
        """Diffuse through all layers, collecting trajectories."""
        trajectories = []
        h = x
        for layer in self.layers:
            traj = layer.diffuse(h, n_steps)
            trajectories.append(traj)
            h = traj[-1]  # Pass final state to next layer
        return trajectories

    def compute_gradients(self, trajectories, class_idx):
        """Local gradient computation per layer."""
        grads = []
        for k, (layer, traj) in enumerate(zip(self.layers, trajectories)):
            residual = layer.compute_residual(traj, class_idx)
            grad_bias = -residual.mean(axis=0) / (2 * self.physics.kT)
            grads.append(grad_bias)
        return grads
```

### Milestones

| Phase | Goal | Metric |
|-------|------|--------|
| 1 | 2-layer network on MNIST | >80% accuracy |
| 2 | 3+ layers, competitive with MLP | >90% accuracy |
| 3 | Hardware validation on FPGA | Matching simulation |
| 4 | Benchmark vs EP, Contrastive Divergence | Speed/accuracy tradeoffs |

---

## 4. Intellectual Property: Should We Patent?

### Patent Landscape Analysis

#### Extropic's Patent (Application 20250165761)
**"Self-learning Thermodynamic Computing"**
- **Inventors:** Christopher Chamberland, Guillaume Verdon-Akzam
- **Key Claims:** Langevin dynamics for learning weights/biases
- **Architecture:** Clamped + unclamped + server thermodynamic chips
- **Status:** Application (not yet granted)

**Differentiation from our approach:**
| Aspect | Extropic | Our Approach |
|--------|----------|--------------|
| Learning mechanism | Clamped/unclamped comparison | Single-pass trajectory observation |
| Gradient source | Chip-to-chip correlation | Onsager-Machlup action |
| Hardware requirement | Multiple synchronized chips | Single device |
| Training phases | Two (like EP) | One |

#### Prior Art Summary

| Work | Year | Relevance |
|------|------|-----------|
| Contrastive Divergence (Hinton) | 2002 | RBM training, two-phase |
| Equilibrium Propagation | 2017 | Energy-based, two-phase, local |
| Stochastic Thermodynamics of Learning | 2017 | Thermodynamic bounds on learning |
| Quantum EP with Onsager reciprocity | 2025 | Connects Onsager to EP |
| Onsager-Machlup + diffusion models | 2025 | OM for transition path sampling |

### Our Novel Contributions

1. **Onsager-Machlup action as training objective** for discriminative learning
2. **Single-pass analytical gradients** without equilibration phases
3. **Trajectory residual formula:** `∂L/∂b = -residual / (2kT)`
4. **φ⁴ + bias potential** for classification (bistable attractor basins)
5. **Conservative pixel diffusion** as data-dependent noise process

### Patentability Assessment

**Strengths:**
- Specific, non-obvious gradient formulas
- Hardware-friendly (local, single-pass)
- Demonstrated experimental results (76% MNIST)
- Distinct from Extropic's clamped/unclamped approach

**Risks:**
- Onsager-Machlup is classical physics (1953)
- Trajectory-based learning is known in physics
- Could be argued as "applying known physics to ML"
- Extropic's broad claims may overlap

### Recommended Actions

#### Option A: Provisional Patent (Recommended)
**Cost: $1,500-3,000 with attorney | Timeline: 2-4 weeks**

File a provisional patent application covering:
1. Trajectory-based gradient estimation using Onsager-Machlup action
2. Single-pass learning without equilibration phases
3. φ⁴ + bias potential architecture for classification
4. Conservative diffusion process for learning

**Benefits:**
- Establishes priority date
- 12 months to file full application
- "Patent pending" status
- Low cost compared to full patent

#### Option B: Publish First (Defensive)
**Cost: $0 | Timeline: Immediate**

Publish paper on arXiv immediately to establish prior art.

**Benefits:**
- Prevents others from patenting
- Academic recognition
- No legal costs

**Risks:**
- Cannot patent later (1-year grace period in US only)
- Competitors can use freely

#### Option C: Trade Secret
Keep implementation details proprietary while publishing theory.

### Provisional Patent Outline

```
Title: Method and System for Training Energy-Based Models
       Using Trajectory-Based Gradient Estimation

Claims:
1. A method for training an energy-based model comprising:
   a) Observing a stochastic trajectory of states under diffusion
   b) Computing residuals between observed and predicted displacements
   c) Estimating gradients using the formula: ∂L/∂θ = f(residual, kT)
   d) Updating model parameters using said gradients

2. The method of claim 1, wherein the gradient formula is derived
   from the Onsager-Machlup action functional.

3. The method of claim 1, wherein the energy function comprises
   a bistable potential with class-specific biases.

4. A thermodynamic computing device configured to perform the
   method of claim 1 using physical stochastic dynamics.
```

---

## Next Steps (Prioritized)

### Immediate (This Week)
- [ ] Implement thrml/thermox integration for validation
- [ ] Draft 2-page provisional patent summary
- [ ] Order Zybo Z7-20 development board

### Short-term (2-4 Weeks)
- [ ] Prototype 2-layer stacked architecture
- [ ] Complete FPGA p-bit array (10x10 minimum)
- [ ] Consult patent attorney for provisional filing

### Medium-term (1-3 Months)
- [ ] Benchmark stacked model vs EP and Contrastive Divergence
- [ ] Hardware validation: run trained model on FPGA
- [ ] Submit paper to NeurIPS/ICML workshop

---

## References

### Hardware
- [FPGA_SC Library](https://github.com/hinata9276/FPGA_SC)
- [pc-COP: 2048 p-bit accelerator](https://arxiv.org/html/2504.04543v1)
- [CMOS p-bit with bistable resistor](https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/adfm.202307935)

### Simulators
- [thermox (Normal Computing)](https://github.com/normal-computing/thermox)
- [thrml (Extropic)](https://github.com/extropic-ai/thrml)

### Deep Architectures
- [Equilibrium Propagation](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full)
- [Deep Boltzmann on Ising machines](https://www.nature.com/articles/s41928-024-01182-4)
- [Quantum EP with Onsager](https://www.nature.com/articles/s41467-025-61665-6)

### Patents
- [Extropic 20250165761](https://patents.justia.com/patent/20250165761)
- [Onsager-Machlup + diffusion](https://arxiv.org/abs/2504.18506)
