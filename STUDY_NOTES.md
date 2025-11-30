# Criticality Engine: Study Notes

## What Works

### 1. φ⁴ + Ising Denoising ✅
The paper's ansatz (Eq. 4):
```
∂_i V = 2J₂x_i + 4J₄x_i³ + b_i + Σⱼ Jᵢⱼxⱼ
```

With J₂ < 0, J₄ > 0 (Higgs/double-well):
- Each pixel has bistable states (±1 attractors)
- Couplings J_ij coordinate neighboring pixels
- **Denoising works**: noisy digit → clean, sharp digit
- Score alignment ~0.8 (learned drift points toward data)

### 2. RBM Generation ✅
Standard RBM with contrastive divergence:
- 500 hidden units, 14×14 visible
- **Generation works**: thermal equilibrium produces recognizable digits
- **Denoising works**: relaxation cleans corrupted inputs
- Reconstruction error: 0.016 (very good)

---

## What Doesn't Work

### 1. φ⁴ + Ising Generation ❌
Same model that denoises well fails at generation:
- Random init → sparse blobs, not digits
- Annealing (T: 5→0.1) doesn't help much
- System finds spurious local minima

### 2. Pure Quadratic Potential ❌
V(x) = ½||x||² + b·x + ½x^T J x (no x⁴ term):
- Creates Gaussian (unimodal) distribution
- Collapses to mean during denoising
- Cannot represent multi-modal digit distribution

---

## Why RBM Works But Our Ansatz Doesn't (for Generation)

### The Key Difference: Training Objective

**RBM + Contrastive Divergence:**
```
ΔW ∝ ⟨v_i h_j⟩_data - ⟨v_i h_j⟩_model
```
- Explicitly samples from model distribution (negative phase)
- Shapes the GLOBAL energy landscape
- Hidden units create multiple modes automatically

**Our Ansatz + Denoising Score Matching:**
```
Loss = ||∇V_model(x̃) - ∇V_true(x̃)||²
```
where x̃ = x_data + noise

- Only matches LOCAL gradients near data points
- Doesn't constrain energy far from data
- No explicit model sampling → spurious minima persist

### Detailed Balance Consideration

Both systems satisfy detailed balance at equilibrium:
```
p(x)P(x→x') = p(x')P(x'→x)
```

But detailed balance only guarantees convergence to equilibrium **if you can reach it**. The training objectives differ in what they optimize:

| Aspect | RBM | Our Ansatz |
|--------|-----|------------|
| Training | Global (samples from model) | Local (gradients near data) |
| Hidden units | Yes (create modes) | No (J_ij is visible-visible) |
| Generation | ✅ Finds modes | ❌ Stuck in spurious minima |
| Denoising | ✅ | ✅ |

### The Missing Piece

Our J_ij = W W^T is low-rank coupling between **visible units only**. This creates a single "basin" in the energy landscape (possibly with some structure from φ⁴).

RBM's W couples visible to **hidden units**. The hidden units act as "mode selectors" - each configuration of h activates a different visible pattern.

---

## The Quantum-Classical Correspondence

d-dim Langevin ↔ (d+1)-dim classical field theory

Our 196-pixel system with time evolution is a 197-dimensional field theory:
- x(t) = worldline / field configuration
- S[x] = ∫ dt (ẋ + ∇V)² / 4μkT (Onsager-Machlup action)
- P[trajectory] ∝ exp(-S)

At criticality (transistors at threshold):
- Long-range temporal correlations
- Efficient exploration of configuration space
- τ_relax ~ ξ^z (dynamical critical exponent)

---

## New Hypothesis: Class-Conditional Latent Modulation

**Problem**: Our model learns one potential for all digits. Generation fails because it can't find the specific digit manifolds.

**Idea**: Learn digit-specific modulation of the potential.

Like NMF: M = W H where H encodes "which digit"
- For each class k, learn a bias vector b_k or coupling modulation
- During inference: specify class → activate class-specific potential
- Generation becomes: "sample from the 3-potential"

### Implementation Plan

1. Learn per-class biases: {b_0, b_1, ..., b_9}
2. Training: for (x, label), use b_label when computing gradients
3. Generation: specify label → use b_label → sample
4. Denoising: try all b_k, pick the one with lowest energy

This is analogous to:
- Conditional Boltzmann machines
- Class-conditional energy models
- Hopfield networks with stored patterns

---

## Experiment Results

### Class-Conditional φ⁴ + Ising ✅ **SUCCESS**

Per-digit bias vectors b_k create digit-specific attractors.

**Results:**
1. **Learned biases show digit templates** - b_k looks like digit k
2. **Conditional generation improved** - 0,4,6,7,8 recognizable
3. **Conditional denoising with correct label** - excellent reconstruction
4. **Conditional denoising with wrong label** - reshapes toward wrong digit!

**Key insight**: Class labels are **latent mode selectors**:
- Unconditional: one shared potential → collapses to mean
- Conditional: per-class potential → digit-specific attractors

This is the NMF / Hopfield / conditional energy model connection:
- NMF: M = WH where H selects basis
- Hopfield: stored patterns as attractors
- Ours: b_k selects which digit attractor to use

---

## Key Clarification: RBM vs Our Model

### Why RBM generates unconditionally but we can't:

**RBM has hidden units that create implicit modes:**
```
p(v) = Σ_h exp(-E(v,h)) / Z  ← mixture over h
```
Each h configuration activates a different pattern. The sum creates multiple modes automatically.

**Our model has no hidden units:**
```
p(x) = exp(-V(x)) / Z  ← single potential
```
The φ⁴ creates per-pixel bistability, but doesn't organize pixels into coherent class patterns.

### But RBM also can't target specific classes!

| Task | RBM (no labels) | Our φ⁴ + b_k |
|------|-----------------|--------------|
| Generate random digit | ✅ | ❌ |
| Generate SPECIFIC digit | ❌ | ✅ |
| Denoise → nearest class | ✅ | ❌ |
| Denoise → SPECIFIC class | ❌ | ✅ |

### The insight:
- **RBM**: Modes emerge implicitly from h, but you can't control which mode
- **Our b_k**: Modes are explicit, and you directly control which attractor is active

Both need class conditioning for targeted generation. The difference is WHERE the mode structure lives:
- RBM: In the hidden layer (learned implicitly)
- Ours: In the bias banks (specified explicitly)

### Hardware implication:
Our explicit b_k approach is actually more hardware-friendly for targeted tasks:
- Load bias bank k → generate/denoise digit k
- No need to search over hidden configurations
- Direct control via reconfiguration fabric

---

## Gemini 3 Analysis: Critical Evaluation

### What Gemini Got Right ✅

1. **Velocity matching = our gradient formula**
   - Eq 7-8: ΔJ ∝ (f(x) - ẋ) · x
   - This IS denoising score matching in physical form

2. **Time-reversal story**
   - Forward (data→noise) = relaxation
   - Backward (noise→data) = reconstruction
   - Same dynamics, different direction

3. **Transistor as Gibbs sampler**
   - Sub-threshold I-V: I_ds = I_0 exp(κ(V_gs - V_th))
   - Implements Boltzmann factor physically

### What Gemini Got Wrong ❌

1. **"Unconditional generation works"** - NO!
   - We validated: unconditional fails completely
   - Local score matching ≠ global mode learning
   - Gemini missed the mode collapse problem

2. **"Physical diffusion model"** - Partially wrong
   - Not like DDPM (no noise schedule, no multiple scales)
   - More like single-scale energy-based model
   - The analogy is loose, not exact

3. **Overstated novelty on transistor physics**
   - Standard p-bit literature (Camsari et al.)
   - Not a new insight, just good pedagogy

### Our Validated Findings (Not in Gemini)

1. **φ⁴ term is essential** - quadratic collapses to mean
2. **Class-conditional biases solve generation**
3. **Attractors visualizable** - gray → digit under V_k
4. **Wrong-label test** - reshapes toward wrong class
5. **Hardware mapping** - bias banks for class selection

---

## Synthesis: Paper Additions

Added to LaTeX:
1. **Velocity matching subsection** (Sec 3.3)
2. **Time-reversal subsection** (Sec 3.4)
3. **Critical evaluation** of score matching limitations
4. **Expanded experiments** with failure modes documented
5. **Attractor visualization** results
6. **Hardware interpretation** of class-conditional biases

---

## Remaining Questions

1. **RBM with paper's learning rule**
   - Use trajectory likelihood (Eqs 7-8) instead of CD
   - See if it still generates well

2. **Full unconditional generation**
   - Need contrastive training (sample from model)?
   - Or learn a "mixture" potential that has all modes?

3. **Multi-scale approach**
   - Like diffusion models: multiple noise levels?
   - Could fix unconditional generation
