"""
Multi-dimensional verification - biases only, then add couplings.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

N = 5  # nodes
mu = 1.0
kT = 1.0
dt = 0.01
n_steps = 100
n_replicas = 100
n_epochs = 300
lr = 1.0

# Target biases
target_b = np.array([0.5, -0.3, 0.2, -0.1, 0.4])

# Potential: V = 0.5 * sum(x_i^2) + sum(b_i * x_i)
# So ∂V/∂x_i = x_i + b_i
# Equilibrium at x_i = -b_i

def langevin_step(x, b):
    """One step with V = 0.5*|x|^2 + b·x"""
    grad_V = x + b
    noise = np.random.randn(N)
    dx = -mu * grad_V * dt + np.sqrt(2 * kT * mu * dt) * noise
    return x + dx


# Initialize learned biases
learned_b = np.zeros(N)

print(f"Target b = {target_b}")
print(f"Initial learned b = {learned_b}")
print()

errors = []

for epoch in range(n_epochs):
    total_grad = np.zeros(N)
    count = 0

    for r in range(n_replicas):
        x = np.random.randn(N) * 0.5
        for k in range(n_steps):
            x_k = x.copy()
            x = langevin_step(x, target_b)  # Step with TARGET
            x_kp1 = x.copy()

            # Compute gradient under LEARNED model
            dx = x_kp1 - x_k
            grad_V_learned = x_k + learned_b
            expected_dx_learned = -mu * grad_V_learned * dt
            residual = dx - expected_dx_learned

            # ∂(log P)/∂b_i = -residual_i / (2kT)
            grad = -residual / (2 * kT)
            total_grad += grad
            count += 1

    avg_grad = total_grad / count

    # Gradient ASCENT on log P
    learned_b = learned_b + lr * avg_grad

    error = np.linalg.norm(learned_b - target_b)
    errors.append(error)

    if epoch % 30 == 0:
        print(f"Epoch {epoch}: error = {error:.4f}")

print()
print(f"Final learned b = {learned_b}")
print(f"Target b        = {target_b}")
print(f"Final error = {errors[-1]:.4f}")

# Plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(errors)
plt.xlabel('Epoch')
plt.ylabel('||learned_b - target_b||')
plt.title('Bias Learning Convergence')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
x = np.arange(N)
width = 0.35
plt.bar(x - width/2, target_b, width, label='Target', alpha=0.8)
plt.bar(x + width/2, learned_b, width, label='Learned', alpha=0.8)
plt.xlabel('Node')
plt.ylabel('Bias value')
plt.title('Bias Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('verify_multi.png', dpi=150)
print("\nSaved to verify_multi.png")
