"""
Full verification - biases AND couplings.
This is the real test of the Criticality Engine learning rules.
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
n_replicas = 200
n_epochs = 500
lr_b = 1.0
lr_J = 0.5

# Target parameters
target_b = np.array([0.3, -0.2, 0.1, -0.1, 0.2])

# Sparse symmetric couplings
target_J = np.zeros((N, N))
target_J[0, 1] = target_J[1, 0] = 0.15
target_J[1, 2] = target_J[2, 1] = -0.1
target_J[2, 3] = target_J[3, 2] = 0.2
target_J[3, 4] = target_J[4, 3] = -0.15
target_J[0, 4] = target_J[4, 0] = 0.1

# Potential: V = 0.5 * sum(x_i^2) + sum(b_i * x_i) + 0.5 * sum_{i<j} J_ij * x_i * x_j
# So ∂V/∂x_i = x_i + b_i + sum_j J_ij x_j

def potential_grad(x, b, J):
    return x + b + J @ x

def langevin_step(x, b, J):
    grad_V = potential_grad(x, b, J)
    noise = np.random.randn(N)
    dx = -mu * grad_V * dt + np.sqrt(2 * kT * mu * dt) * noise
    return x + dx


# Initialize learned parameters
learned_b = np.zeros(N)
learned_J = np.zeros((N, N))

print(f"Target b = {target_b}")
print(f"Target J (non-zero):")
for i in range(N):
    for j in range(i+1, N):
        if target_J[i,j] != 0:
            print(f"  J[{i},{j}] = {target_J[i,j]:.2f}")
print()

errors_b = []
errors_J = []

for epoch in range(n_epochs):
    total_grad_b = np.zeros(N)
    total_grad_J = np.zeros((N, N))
    count = 0

    for r in range(n_replicas):
        x = np.random.randn(N) * 0.5
        for k in range(n_steps):
            x_k = x.copy()
            x = langevin_step(x, target_b, target_J)  # Step with TARGET
            x_kp1 = x.copy()

            # Compute gradient under LEARNED model
            dx = x_kp1 - x_k
            grad_V_learned = potential_grad(x_k, learned_b, learned_J)
            expected_dx_learned = -mu * grad_V_learned * dt
            residual = dx - expected_dx_learned

            # Gradients (from paper Eqs 9-10, adjusted for our potential form)
            # ∂(log P)/∂b_i = -residual_i / (2kT)
            grad_b = -residual / (2 * kT)

            # ∂(log P)/∂J_ij = -residual_i * x_j / (2kT) - residual_j * x_i / (2kT)
            # (for symmetric J)
            grad_J = -np.outer(residual, x_k) / (2 * kT)
            grad_J = grad_J + grad_J.T  # Symmetrize

            total_grad_b += grad_b
            total_grad_J += grad_J
            count += 1

    avg_grad_b = total_grad_b / count
    avg_grad_J = total_grad_J / count

    # Zero diagonal of J gradient
    np.fill_diagonal(avg_grad_J, 0)

    # Gradient ASCENT on log P
    learned_b = learned_b + lr_b * avg_grad_b
    learned_J = learned_J + lr_J * avg_grad_J

    # Keep J symmetric
    learned_J = 0.5 * (learned_J + learned_J.T)
    np.fill_diagonal(learned_J, 0)

    error_b = np.linalg.norm(learned_b - target_b)
    error_J = np.linalg.norm(learned_J - target_J)
    errors_b.append(error_b)
    errors_J.append(error_J)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: error_b = {error_b:.4f}, error_J = {error_J:.4f}")

print()
print(f"Final learned b = {learned_b}")
print(f"Target b        = {target_b}")
print(f"Final error_b = {errors_b[-1]:.4f}")
print()
print(f"Learned J (non-zero entries > 0.01):")
for i in range(N):
    for j in range(i+1, N):
        if abs(learned_J[i,j]) > 0.01 or abs(target_J[i,j]) > 0.01:
            print(f"  J[{i},{j}]: learned={learned_J[i,j]:.3f}, target={target_J[i,j]:.3f}")
print(f"Final error_J = {errors_J[-1]:.4f}")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(errors_b)
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('||learned_b - target_b||')
axes[0,0].set_title('Bias Learning')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(errors_J)
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('||learned_J - target_J||')
axes[0,1].set_title('Coupling Learning')
axes[0,1].grid(True, alpha=0.3)

x = np.arange(N)
width = 0.35
axes[1,0].bar(x - width/2, target_b, width, label='Target', alpha=0.8)
axes[1,0].bar(x + width/2, learned_b, width, label='Learned', alpha=0.8)
axes[1,0].set_xlabel('Node')
axes[1,0].set_ylabel('Bias')
axes[1,0].set_title('Bias Comparison')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Show J matrices
im = axes[1,1].imshow(np.abs(learned_J - target_J), cmap='Reds')
axes[1,1].set_title('|learned_J - target_J|')
axes[1,1].set_xlabel('Node j')
axes[1,1].set_ylabel('Node i')
plt.colorbar(im, ax=axes[1,1])

plt.tight_layout()
plt.savefig('verify_full.png', dpi=150)
print("\nSaved to verify_full.png")
