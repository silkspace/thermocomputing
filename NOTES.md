# Notes: Criticality Engine Architecture

## Core Insight

The paper's premise rests on a beautiful unification:

**Criticality operates at two levels:**

1. **Device level**: A CMOS transistor biased at threshold IS a critical system. The transfer function σ(h) = 1/(1+e^{-h}) is the order parameter of a system at its critical point. Maximum susceptibility ∂σ/∂h peaks at h=0. The p-bit naturally samples from p(s=1) = σ(h) — this is Gibbs sampling baked into physics.

2. **Learning level**: Finding θ* that minimizes KL divergence to data = tuning the model toward criticality w.r.t. the target distribution. A well-trained generative model is "critical" in the sense of being maximally sensitive to the structure in the data.

The transistor is not *simulating* criticality. It *is* critical. This is the hardware advantage.

---

## What the Architecture Actually Is

### The Criticality Engine (née MWP) has five layers:

**1. T-array (Thermodynamic Array)**
- N stochastic nodes (p-bits or oscillators)
- Each node i has state x_i(t) evolving under Langevin dynamics:
  ```
  dx_i/dt = f_i(x; θ) + √(2D) η_i(t)
  ```
- Parameters θ = {h_i, J_ij} implemented as physical currents/conductances
- Replicated R times for trajectory ensembles

**2. Measurement Plane**
- High-impedance sampling network
- Global SAMPLE signal at configurable times t_k
- Sample-and-hold circuits latch states into registers
- Minimal perturbation (capacitive coupling)

**3. Local Statistics Layer**
- For each parameter θ_α, a correlator block computes:
  ```
  m_i(t_k) = (1/R) Σ_r x_i^(r)(t_k)
  C_ij(t_k) = (1/R) Σ_r x_i^(r)(t_k) x_j^(r)(t_k)
  ```
- These are sufficient statistics for gradient estimation
- Can be analog accumulators or small digital MACs

**4. Gradient Engine**
- Lightweight digital core (vector microcontroller)
- Reads correlator registers
- Applies learning rule (see below)
- Writes updates to shadow parameter bank

**5. Reconfiguration Fabric**
- Dual-bank weight memory (active/shadow)
- DAC arrays convert digital weights → analog currents
- Current mirrors implement couplings: I_{j←i} = J_ij · I_i
- Global bank-select flip for atomic updates

---

## How Current Mirrors Implement Weights

This is the key hardware primitive. A current mirror:
- Takes reference current I_ref
- Produces scaled copy I_out = k · I_ref
- Two matched MOSFETs with tied gates

For p-bit coupling:
1. p-bit i outputs fluctuating current representing its state
2. Current mirror scales it by J_ij (set by DAC)
3. Scaled current injected into p-bit j's bias input
4. Thus: influence of i→j is physical current ∝ J_ij

Negative couplings: differential pairs or push-pull configurations.

---

## Learning Rules

### Finite-time objective at observation time t_k:
```
L(θ) = E_{x(t_k) ~ p_θ^(k)} [ℓ(x(t_k), y)]
```

### Gradient estimation options:

**1. Likelihood-ratio (score function)**
```
∇_θ L = E[ℓ(x_k, y) · ∇_θ log p_θ(x_k)]
```
For Ising/p-bit: ∂log p / ∂J_ij ∝ s_i s_j - ⟨s_i s_j⟩_θ

**2. Contrastive (Boltzmann-style)**
```
ΔJ_ij ∝ C_ij^target(t_k) - C_ij^model(t_k)
```
Computed directly from correlator blocks.

**3. Pathwise (requires noise seeds)**
Store noise realizations, compute ∂x_k/∂θ via adjoint.
More complex but lower variance.

The beauty: all these reduce to local functions of measured statistics {s_i(t_k), s_i s_j(t_k)}.

---

## The Training Loop

1. Initialize parameters θ in active bank
2. Run T-array for time t_k (R replicas, different noise seeds)
3. Assert SAMPLE signal → latch states
4. Correlator blocks compute statistics
5. Gradient engine reads stats, computes Δθ
6. Write θ' = θ - η·Δθ to shadow bank
7. Flip bank-select (atomic parameter update)
8. Repeat

After sufficiently many t_relaxation intervals, we converge toward the minimum.

---

## Why This Matters

**Digital training**: Simulate SDE → numerical integration → backprop through time → update weights. Enormous computational overhead.

**Criticality Engine**: Physics runs the SDE. Measurement gives trajectory samples. Local correlators give gradients. Current mirrors implement weights. The chip trains itself.

Energy advantage: operating near Landauer limit (~kT ln 2 per operation) vs. digital (~10^6 × that).

---

## What's Missing From Current Draft

1. **The criticality story**: Why p-bits at threshold are critical systems. Why this matters for learning (maximal sensitivity, information propagation).

2. **Detailed architecture**: The five-layer stack isn't explicit. Need T-array, measurement plane, statistics layer, gradient engine, reconfiguration fabric.

3. **Learning rule derivations**: Currently hand-wavy. Need at least one rigorous derivation.

4. **Hardware specifics**: Current mirrors, DAC precision, negative couplings, scale estimates.

5. **Connection to continuous-time RNNs**: The mapping f(x) ↔ hidden-state update. This is in the draft but underemphasized.

6. **Energy analysis**: Claims "orders of magnitude" but no numbers.

7. **Related work**: Zero citations. Need Boltzmann machines, equilibrium propagation, Ising machines, Extropic, Normal Computing.

---

## The Name

"Criticality Engine" captures both levels:
- Device criticality (transistors at threshold)
- Learning criticality (model tuned to data)

It's evocative and earnable. The paper should make the reader understand why this name is precise, not just catchy.
