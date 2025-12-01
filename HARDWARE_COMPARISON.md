# Hardware Feasibility: EP vs Onsager-Machlup

## Quick Verdict

**Our approach is significantly more hardware-friendly.** Here's why:

---

## 1. Timing & Control Complexity

### Equilibrium Propagation
```
Phase 1: Initialize → Relax → Detect equilibrium → Store state
Phase 2: Apply nudge → Relax → Detect equilibrium → Compute gradient
```

**Problems:**
- How do you detect equilibrium? Need convergence criterion
- Must store full state vector from phase 1
- Nudge magnitude β must be small but measurable
- Two sequential phases = 2× the time

### Our Approach
```
Initialize → Sample trajectory at fixed intervals → Compute gradients (pipelined)
```

**Advantages:**
- No equilibrium detection needed
- Fixed sampling schedule (simple timer)
- Can compute gradients while trajectory continues
- Single phase = simpler control

**Winner: Ours** (dramatically simpler control logic)

---

## 2. Equilibration Time Problem

This is EP's Achilles heel.

### The Physics
For a system of N coupled units:
- Correlation time τ_corr ~ 1/λ_min (smallest eigenvalue of coupling matrix)
- Equilibration requires ~10-100× τ_corr
- For strongly coupled systems, λ_min can be very small
- **Equilibration time scales poorly with system size**

### Concrete Estimate (N=1000 p-bit array)
```
P-bit flip rate: ~10 MHz (100 ns per flip)
Correlation time: ~100-1000 flips = 10-100 μs
Equilibration: ~10× correlation = 100-1000 μs per phase
Total EP cycle: ~200-2000 μs for one gradient
```

### Our Approach
```
Trajectory length: K=20 steps
Step time: ~10-100 flips = 1-10 μs
Total trajectory: ~20-200 μs
Gradient available: immediately after trajectory
```

**Speed advantage: 2-10× faster per learning step**

---

## 3. The Nudging Problem

EP requires injecting a "nudge" signal:
```
du/dt = -∂E/∂u - β·∂C/∂u
```

### Hardware Implications

To compute β·∂C/∂u, you need:
1. Access to target label (must be available on-chip)
2. Computation of cost gradient (another circuit)
3. Scaling by β (tunable gain)
4. Injection into output units only (selective addressing)

**This requires:**
- Additional DACs for nudge signals
- Multiplexing to target output units
- Precise analog scaling (β must be small but measurable)

### Our Approach: No Nudging

We just **observe** the natural dynamics. The system evolves under:
```
dx = -μ∇V dt + noise
```

No external signal injection during trajectory. The gradient comes from **passive observation**.

**Hardware simplification:**
- No nudge DACs
- No target injection path
- No precision β scaling
- Just sample-and-hold circuits

**Winner: Ours** (eliminates entire nudge subsystem)

---

## 4. Memory Requirements

### EP
Must store:
1. Full state vector from free phase: N values
2. Full state vector from nudged phase: N values
3. Compute difference: N subtractions

For N=1000, 16-bit precision: 4 KB just for state storage

### Our Approach
Must store:
1. Current state: N values
2. Previous state (for Δx): N values
3. Running gradient accumulator: N values (for biases)

Same order of magnitude, but:
- Can stream: don't need to store full trajectory
- Can compute gradients incrementally (reduce, not store)

**Roughly equal**, slight edge to us for streaming capability

---

## 5. Noise Sensitivity

### EP
Gradient = (state_nudged - state_free) / β

**Problem:** Both states are noisy (thermal fluctuations). The difference of two noisy quantities has **higher relative noise**:
```
SNR(A - B) < SNR(A) when A ≈ B
```

For small β (required for accurate gradients), the signal (state difference) is small but noise is unchanged.

**Must average over many repetitions to get clean gradients.**

### Our Approach
Gradient = Σₖ residualₖ / (2kT)

We're **summing** residuals over K trajectory steps. This is natural averaging:
```
SNR improves as √K
```

The noise is part of the physics (it's what drives the diffusion), and we're measuring **deviations from expected drift**, not small differences between equilibria.

**Winner: Ours** (inherent noise averaging along trajectory)

---

## 6. Parallelism & Pipelining

### EP
Sequential phases prevent pipelining:
```
[Free phase] → wait → [Store] → [Nudged phase] → wait → [Compute]
```

Can't start next sample until current gradient is computed.

### Our Approach
Fully pipelined:
```
Time 0: Sample x(t₀)
Time 1: Sample x(t₁), compute residual₀
Time 2: Sample x(t₂), compute residual₁
...
Time K: Compute final gradient, START NEXT SAMPLE
```

While computing gradient for sample n, can already be observing trajectory for sample n+1.

**Winner: Ours** (2× throughput from pipelining)

---

## 7. Robustness to Non-Idealities

Real hardware has:
- Device mismatch
- Parasitic capacitances
- Colored noise (not white)
- Temperature drift

### EP Sensitivity
EP assumes system reaches **exact same equilibrium** under same parameters. Device mismatch means:
- Different chips reach different "equilibria"
- Temperature drift changes equilibrium over time
- Comparing states from different moments is noisy

### Our Approach
We use the **surrogate model** interpretation from the paper:
- We don't claim the hardware obeys detailed balance
- We compute gradients **as if** it did
- Deviations appear as structured residuals (diagnostic)
- The formula works even if the model is approximate

**Winner: Ours** (more robust to hardware non-idealities)

---

## 8. What Hardware People Are Building

### Extropic (clamped/unclamped approach)
Their [patent](https://patents.justia.com/patent/20250165761) describes:
- "Clamped" chip (fixed to data) + "unclamped" chip (free)
- Compare statistics between chips
- This is EP-like (two phases, contrastive)

**But:** They need TWO synchronized chips. More hardware, more complexity.

### Normal Computing (SPU)
From their [thermox paper](https://github.com/normal-computing/thermox):
- System evolves to equilibrium
- Sample from equilibrium distribution
- Primarily for inference/sampling, not training

### P-bit Arrays (Purdue, etc.)
- Designed for combinatorial optimization
- Sample from Boltzmann distribution
- Training typically done offline

### Our Approach: Simplest Path to On-Chip Training
- Single chip
- Passive observation
- No equilibration wait
- No nudging mechanism
- Simple sample-and-hold + local compute

---

## 9. Quantitative Comparison

| Metric | EP | Ours | Advantage |
|--------|----|----|-----------|
| Phases per gradient | 2 | 1 | **2× fewer** |
| Equilibration required | Yes (×2) | No | **Eliminates ~90% wait time** |
| Control complexity | High (phase switching, nudge) | Low (timer + sampler) | **Much simpler** |
| Nudge hardware | Required | Not needed | **Eliminates subsystem** |
| Memory (streaming) | Must store full states | Can stream | **Lower peak memory** |
| Noise handling | Difference of noisy states | Sum over trajectory | **Better SNR** |
| Pipelining | Not possible | Natural | **2× throughput** |
| Robustness | Requires true equilibrium | Works with approximate model | **More forgiving** |

---

## 10. Conclusion

**Our Onsager-Machlup trajectory approach is more hardware-friendly because:**

1. **No equilibration** - the biggest practical win
2. **No nudging** - eliminates an entire subsystem
3. **Single phase** - simpler control, faster cycles
4. **Pipelined** - higher throughput
5. **Noise-robust** - averaging built into the math

**EP's advantage:** More theoretical pedigree, proven on deep networks

**Our advantage:** Simpler hardware path, faster training cycles

For a first-generation on-chip learning system, **our approach has lower implementation risk**.

---

## Estimated Hardware Resources

### EP Implementation
- T-array: N p-bits + couplings
- State storage: 2N registers (free + nudged)
- Nudge DACs: M outputs × precision
- Equilibrium detector: comparator + threshold logic
- Phase controller: state machine
- Gradient compute: N subtractions + scaling

### Our Implementation
- T-array: N p-bits + couplings
- Sample-hold: 2N registers (current + previous)
- Timer: simple counter
- Gradient compute: N MACs (multiply-accumulate)

**We eliminate:** Nudge DACs, equilibrium detector, phase controller

**We add:** Nothing significant

---

## Recommendation

**For FPGA prototype:** Our approach. Simpler to implement, faster iteration.

**For ASIC:** Our approach. Less area, lower power, simpler verification.

**For research exploration:** EP has more published results on deep networks; our approach needs stacked-layer validation.

**Bottom line:** If the goal is to build working hardware soon, our approach has a clearer path.
