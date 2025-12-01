# Deep Dive: Equilibrium Propagation vs Our Approach

## Executive Summary

**Are they the same?** No. While both are local learning rules for energy-based models, they use **different mathematical foundations from Onsager's work**:

| Aspect | Equilibrium Propagation | Our Approach |
|--------|------------------------|--------------|
| **Onsager foundation** | Reciprocal Relations (1931) | Onsager-Machlup Functional (1953) |
| **Core object** | Linear response coefficients | Path probability / trajectory likelihood |
| **Learning phases** | Two (free + nudged) | One (observation only) |
| **Equilibration required?** | Yes, in both phases | No |
| **Gradient source** | Difference between equilibria | Velocity residuals along trajectory |

**Our key novelty:** Single-pass trajectory observation with closed-form gradients from the Onsager-Machlup action. No equilibration, no nudging, no two phases.

---

## 1. Equilibrium Propagation (Scellier & Bengio, 2017)

### Mathematical Framework

**Energy function:**
```
E(u) = (1/2)Σᵢuᵢ² - (1/2)Σᵢ≠ⱼ Wᵢⱼρ(uᵢ)ρ(uⱼ) - Σᵢbᵢρ(uᵢ)
```

**Cost function:**
```
C = (1/2)||y - d||²
```

**Total energy with nudging:**
```
F = E + βC
```

### The Two-Phase Algorithm

**Phase 1 (Free phase, β = 0):**
- Network relaxes: `du/dt = -∂E/∂u`
- Reaches equilibrium fixed point `u⁰`
- This is the "prediction" phase

**Phase 2 (Nudged phase, β > 0):**
- Network relaxes: `du/dt = -∂E/∂u - β∂C/∂u`
- Output units "nudged" toward target
- Reaches new equilibrium `uᵝ`

**Gradient formula (β → 0 limit):**
```
∂L/∂Wᵢⱼ = lim_{β→0} (1/β)[ρ(uᵢᵝ)ρ(uⱼᵝ) - ρ(uᵢ⁰)ρ(uⱼ⁰)]
```

### Connection to Onsager Reciprocity

The 2025 paper "[Quantum EP based on Onsager reciprocity](https://www.nature.com/articles/s41467-025-61665-6)" proves:

> "Classical Onsager reciprocity is equivalent to what has been termed the 'Fundamental Lemma' for classical Equilibrium Propagation."

**Onsager Reciprocal Relations (1931):** The susceptibility matrix is symmetric: χᵢⱼ = χⱼᵢ

This means: perturbing parameter j affects observable i the same way that perturbing parameter i affects observable j.

EP exploits this by:
1. Nudging the **output** (perturb one parameter)
2. Measuring changes in **hidden states** (observe all responses)
3. Reciprocity guarantees this gives the same gradients as perturbing each hidden parameter individually

### Critical Limitation: Requires Equilibrium

EP requires the system to **reach equilibrium twice**:
- Once for free phase (prediction)
- Once for nudged phase (gradient computation)

The paper acknowledges:
> "The experiments show that the free phase is fairly long when performed with a discrete-time computer simulation. However, the full potential of the proposed framework could be exploited on analog hardware."

---

## 2. Our Approach: Onsager-Machlup Action

### Mathematical Framework

**Energy function (φ⁴ + bias):**
```
V(x) = J₂||x||² + J₄||x||⁴ + b·x + Σᵢⱼ Jᵢⱼxᵢxⱼ
```

**Langevin dynamics:**
```
dx = -μ∇V dt + √(2kT μ) dW
```

**Onsager-Machlup action (path probability):**
```
S[x(t)] = ∫₀ᵀ (ẋ + μ∇V)² / (4μkT) dt

P[trajectory] ∝ exp(-S)
```

### Single-Phase Algorithm

**No phases. Just observation:**
1. Start from data x₀
2. Let system diffuse naturally (no control)
3. Observe trajectory {x(t₀), x(t₁), ..., x(tₖ)}
4. Compute gradients from velocity residuals

**Gradient formulas (closed-form):**

For bias bᵢ:
```
∂L/∂bᵢ = (-Δxᵢ + μ·∂ᵢV·Δt) / (2kT)
       = residualᵢ / (2kT)
```

For coupling Jᵢⱼ:
```
∂L/∂Jᵢⱼ = (residualᵢ · xⱼ + residualⱼ · xᵢ) / (2kT)
```

### What is the Onsager-Machlup Functional?

**Onsager-Machlup (1953):** The probability density for a stochastic trajectory.

From [Wikipedia](https://en.wikipedia.org/wiki/Onsager–Machlup_function):
> "The Onsager–Machlup function is used to define a probability density for a stochastic process, and it is similar to the Lagrangian of a dynamical system."

Key properties:
- Gives the **most probable path** between two states
- Minimizing S[x(t)] yields the **expected trajectory**
- When parameters are correct, observed paths have **minimal action**

### No Equilibration Required

We do **not** wait for equilibrium. We observe **finite-time trajectories** and extract gradients from the velocity mismatch:

```
observed_velocity = Δx / Δt
predicted_velocity = -μ∇V
residual = observed - predicted
gradient = residual / (2kT)
```

The system never needs to reach a fixed point. Even short trajectories (5-20 steps) provide useful gradient signal.

---

## 3. Key Differences

### 3.1 Different Onsager Concepts

| Onsager Reciprocal Relations (1931) | Onsager-Machlup Functional (1953) |
|-------------------------------------|-----------------------------------|
| Symmetry of transport coefficients | Path probability for trajectories |
| Linear response at equilibrium | Action functional for dynamics |
| Lᵢⱼ = Lⱼᵢ | S[x(t)] = ∫ (ẋ + μ∇V)² dt |
| Used by EP | Used by us |

These are **different mathematical objects** from the same physicist!

### 3.2 Equilibrium vs Trajectory

**EP:** Compare two equilibrium states (free and nudged)
```
gradient ∝ (equilibrium_nudged - equilibrium_free) / β
```

**Us:** Observe trajectory, compute velocity residuals
```
gradient ∝ Σₖ (velocity_observed - velocity_predicted) / (2kT)
```

### 3.3 Number of Phases

**EP:** Two phases, each requiring equilibration
- Phase 1: Free relaxation → equilibrium
- Phase 2: Nudged relaxation → new equilibrium

**Us:** One phase, no equilibration
- Observe trajectory
- Compute gradients
- Done

### 3.4 Computational Complexity

**EP:** O(τ_eq × 2) per gradient, where τ_eq is equilibration time

**Us:** O(K) per gradient, where K is trajectory length (typically 5-20)

In simulation, EP's "free phase is fairly long" (their words). Our approach works with K=5 trajectory steps.

### 3.5 Hardware Implications

**EP:** Must implement two phases with phase-switching
- Requires ability to "nudge" outputs
- Must wait for equilibration in both phases
- Complex control logic

**Us:** Passive observation only
- No nudging mechanism needed
- No equilibration wait
- Simpler control: just sample and compute

---

## 4. What's Novel in Our Work?

### 4.1 First Use of Onsager-Machlup for Discriminative Learning

Previous uses of Onsager-Machlup in ML:
- [Transition path sampling](https://arxiv.org/abs/2504.18506) (2025) - generative, not discriminative
- Physics simulations - finding most probable paths

Our contribution: Using O-M action to train a **classifier** with per-class biases.

### 4.2 Closed-Form Local Gradients

From the paper (Eq 12-13):
```
∂L/∂bᵢ = residualᵢ / (2kT)
∂L/∂Jᵢⱼ = (residualᵢ·xⱼ + residualⱼ·xᵢ) / (2kT)
```

These are **analytical**, not requiring autodiff or finite differences.

### 4.3 Constraint Satisfaction Interpretation

We frame learning as projecting onto a constraint surface:
```
C = {θ : S[x_obs; θ] = S_thermal}
```

When parameters are correct, observed trajectories have minimal action (just thermal noise). This is **qualitatively different** from EP's contrastive approach.

### 4.4 Temperature as Lagrangian Multiplier

We show that kT controls constraint tightness:
- Low T: Strict constraint, parameters must explain trajectories exactly
- High T: Soft constraint, more tolerance

This Lagrangian interpretation is new.

### 4.5 Single-Epoch MNIST Classification

76% accuracy in one epoch. EP papers typically show multi-epoch convergence. The fast convergence comes from our constraint-satisfaction geometry.

---

## 5. What's NOT Novel

To be intellectually honest:

### 5.1 Local Learning Rules for EBMs
EP, Contrastive Divergence, and Contrastive Hebbian Learning all have local update rules. Locality itself is not new.

### 5.2 Velocity Matching / Score Matching
Our gradient formula resembles denoising score matching. The connection to score functions is known.

### 5.3 Energy-Based Classification
Using energy wells for classification (argmin_c V_c(x)) is standard in EBM literature.

### 5.4 Hardware Thermodynamic Computing
Extropic, Normal Computing, and others are building thermodynamic hardware. The hardware concept isn't ours.

---

## 6. Prior Art Summary

| Work | Year | Relationship to Ours |
|------|------|---------------------|
| Boltzmann Machine | 1985 | Energy-based model, two-phase learning |
| Contrastive Divergence | 2002 | Approximate gradient via short MCMC |
| **Equilibrium Propagation** | 2017 | Two-phase, Onsager reciprocity |
| Quantum EP (Onsager) | 2025 | Connects EP to Onsager reciprocity |
| Stochastic Thermo of Learning | 2017 | Thermodynamic bounds on learning |
| O-M + Diffusion Models | 2025 | O-M for transition paths (generative) |

**Our position:** First to use Onsager-Machlup action for single-pass discriminative learning with closed-form gradients.

---

## 7. Conclusion

### Are we doing what EP does?
**No.**

EP uses Onsager Reciprocity (1931) to compute gradients by comparing two equilibria.

We use the Onsager-Machlup Functional (1953) to compute gradients from trajectory velocity residuals.

### Is our work novel?
**Yes**, in specific ways:
1. First discriminative use of O-M action
2. Single-pass (no equilibration)
3. Closed-form gradient formulas
4. Constraint satisfaction interpretation
5. Temperature as Lagrangian multiplier

### Should we cite EP?
**Yes.** It's the most closely related prior work. We should:
1. Clearly distinguish our mathematical foundation (O-M 1953 vs reciprocity 1931)
2. Highlight the single-phase vs two-phase difference
3. Compare convergence speeds empirically

---

## References

### Equilibrium Propagation
- Scellier, B., & Bengio, Y. (2017). [Equilibrium Propagation: Bridging the Gap between Energy-Based Models and Backpropagation](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full). Frontiers in Computational Neuroscience.

### Quantum EP / Onsager Connection
- Wanjura, C. C., et al. (2025). [Quantum equilibrium propagation for efficient training of quantum systems based on Onsager reciprocity](https://www.nature.com/articles/s41467-025-61665-6). Nature Communications.

### Onsager-Machlup Functional
- Onsager, L., & Machlup, S. (1953). Fluctuations and Irreversible Processes. Physical Review.
- [Wikipedia: Onsager-Machlup function](https://en.wikipedia.org/wiki/Onsager–Machlup_function)

### Unifying Theory
- Millidge, B., et al. (2022). [Backpropagation at the Infinitesimal Inference Limit of Energy-Based Models](https://arxiv.org/abs/2206.02629). arXiv.

### Score Matching Connection
- Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution.
