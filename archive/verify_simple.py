"""
Simplified verification - just learn biases with fixed J2, J4, J.
"""

import numpy as np

np.random.seed(42)

# Simple 1D case first
N = 1
mu = 1.0
kT = 1.0
dt = 0.01
n_steps = 100
n_replicas = 50
n_epochs = 500
lr = 2.0

# Target: just a bias
target_b = 0.5

# Potential: V = 0.5 * x^2 + b*x  (quadratic well shifted by bias)
# So ∂V/∂x = x + b
# Equilibrium at x = -b

def langevin_step(x, b):
    """One step with V = 0.5*x^2 + b*x"""
    grad_V = x + b
    noise = np.random.randn()
    dx = -mu * grad_V * dt + np.sqrt(2 * kT * mu * dt) * noise
    return x + dx

def compute_grad_b(x_k, x_kp1, b):
    """
    Gradient of -log P w.r.t. b.

    P(dx | x) is Gaussian with mean = -μ(x+b)dt, var = 2kTμdt

    log P = -|dx - mean|^2 / (2*var) + const

    ∂(log P)/∂b = (dx - mean) * ∂mean/∂b / var
                = (dx + μ(x+b)dt) * (-μdt) / (2kTμdt)
                = -(dx + μ(x+b)dt) / (2kT)

    So -∂(log P)/∂b = (dx + μ(x+b)dt) / (2kT)

    But we want to MAXIMIZE log P, so we do gradient ascent:
    b ← b + lr * ∂(log P)/∂b = b - lr * (dx + μ(x+b)dt) / (2kT)
    """
    dx = x_kp1 - x_k
    grad_V = x_k + b  # At x_k
    expected_dx = -mu * grad_V * dt
    residual = dx - expected_dx  # observed - expected

    # ∂(log P)/∂b: how does log P change with b?
    # mean = -μ(x+b)dt, so ∂mean/∂b = -μdt
    # ∂(log P)/∂b = residual * ∂mean/∂b / var = residual * (-μdt) / (2kTμdt)
    #             = -residual / (2kT)

    grad_log_P = -residual / (2 * kT)
    return grad_log_P

# Initialize
learned_b = 0.0

print(f"Target b = {target_b}")
print(f"Initial learned b = {learned_b}")
print(f"Target equilibrium x = {-target_b}")
print()

for epoch in range(n_epochs):
    # Generate trajectories from TARGET
    total_grad = 0.0
    count = 0

    for r in range(n_replicas):
        x = np.random.randn() * 0.5  # Random init
        for k in range(n_steps):
            x_k = x
            x = langevin_step(x, target_b)  # Step with TARGET
            x_kp1 = x

            # Compute gradient under LEARNED model
            # But wait - we need grad_V evaluated with learned_b
            dx = x_kp1 - x_k
            grad_V_learned = x_k + learned_b
            expected_dx_learned = -mu * grad_V_learned * dt
            residual = dx - expected_dx_learned

            # ∂(log P)/∂b = -residual / (2kT)
            grad = -residual / (2 * kT)
            total_grad += grad
            count += 1

    avg_grad = total_grad / count

    # Gradient ASCENT on log P
    learned_b = learned_b + lr * avg_grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: learned_b = {learned_b:.4f}, grad = {avg_grad:.6f}")

print()
print(f"Final learned b = {learned_b:.4f}")
print(f"Target b = {target_b:.4f}")
print(f"Error = {abs(learned_b - target_b):.4f}")
