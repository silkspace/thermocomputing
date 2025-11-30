"""
Verification of the Criticality Engine learning rules.

We simulate Langevin dynamics and verify that the gradient estimator
from Eqs 9-13 in the paper actually recovers target parameters.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Params:
    """Learnable parameters of the thermodynamic system."""
    J2: float          # Quadratic self-coupling
    J4: float          # Quartic self-coupling
    b: np.ndarray      # Biases (N,)
    J: np.ndarray      # Pairwise couplings (N, N), symmetric

    def copy(self):
        return Params(
            J2=self.J2,
            J4=self.J4,
            b=self.b.copy(),
            J=self.J.copy()
        )


def potential_gradient(x: np.ndarray, params: Params) -> np.ndarray:
    """
    Compute ∂_i V(x; θ) for each node.

    From Eq 5 in paper:
    ∂_i V = 2*J2*x_i + 4*J4*x_i^3 + b_i + Σ_j J_ij x_j
    """
    grad = (2 * params.J2 * x +
            4 * params.J4 * x**3 +
            params.b +
            params.J @ x)
    return grad


def langevin_step(x: np.ndarray, params: Params,
                  mu: float, kT: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    One Euler-Maruyama step of overdamped Langevin dynamics.

    dx = -μ ∂V/∂x dt + √(2kT μ dt) η

    Returns: (x_new, noise_used)
    """
    grad_V = potential_gradient(x, params)
    noise = np.random.randn(len(x))
    dx = -mu * grad_V * dt + np.sqrt(2 * kT * mu * dt) * noise
    return x + dx, noise


def compute_gradients(x_k: np.ndarray, x_kp1: np.ndarray,
                      params: Params, mu: float, kT: float, dt: float):
    """
    Compute gradients of negative log-likelihood w.r.t. parameters.

    From Eqs 9-10 in paper:

    For J_ij:
    -∂/∂J_ij ln P = [(-Δx_i + μ ∂_i V Δt)/(2kT)] x_j + [(-Δx_j + μ ∂_j V Δt)/(2kT)] x_i

    For b_i:
    -∂/∂b_i ln P = (-Δx_i + μ ∂_i V Δt)/(2kT)
    """
    N = len(x_k)
    dx = x_kp1 - x_k  # Δx^k

    # Evaluate potential gradient at x_k (could also use x_kp1 or midpoint)
    grad_V = potential_gradient(x_k, params)

    # The "residual" term: what the displacement should have been (deterministic part)
    # minus what it actually was
    residual = -dx + mu * grad_V * dt  # This is -Δx + μ∂V Δt
    residual_scaled = residual / (2 * kT)

    # Gradient w.r.t. biases (Eq 10)
    # -∂L/∂b_i = residual_scaled_i * (∂(∂_i V)/∂b_i) = residual_scaled_i * 1
    grad_b = residual_scaled.copy()

    # Gradient w.r.t. J_ij (Eq 9)
    # -∂L/∂J_ij = residual_scaled_i * x_j + residual_scaled_j * x_i
    # (since ∂(∂_i V)/∂J_ij = x_j and ∂(∂_j V)/∂J_ij = x_i)
    grad_J = np.outer(residual_scaled, x_k) + np.outer(x_k, residual_scaled)

    # Gradient w.r.t. J2 and J4
    # ∂(∂_i V)/∂J2 = 2*x_i, so -∂L/∂J2 = Σ_i residual_scaled_i * 2*x_i
    grad_J2 = 2 * np.sum(residual_scaled * x_k)

    # ∂(∂_i V)/∂J4 = 4*x_i^3, so -∂L/∂J4 = Σ_i residual_scaled_i * 4*x_i^3
    grad_J4 = 4 * np.sum(residual_scaled * x_k**3)

    return grad_J2, grad_J4, grad_b, grad_J


def generate_trajectory(x0: np.ndarray, params: Params,
                        mu: float, kT: float, dt: float,
                        n_steps: int) -> np.ndarray:
    """Generate a trajectory of n_steps from initial condition x0."""
    N = len(x0)
    traj = np.zeros((n_steps + 1, N))
    traj[0] = x0

    x = x0.copy()
    for k in range(n_steps):
        x, _ = langevin_step(x, params, mu, kT, dt)
        traj[k + 1] = x

    return traj


def train_step(trajectories: np.ndarray, params: Params,
               mu: float, kT: float, dt: float, lr: float) -> Params:
    """
    One training step: compute gradients over trajectories and update params.

    trajectories: (n_replicas, n_steps+1, N) array
    """
    n_replicas, n_steps_plus_1, N = trajectories.shape
    n_steps = n_steps_plus_1 - 1

    # Accumulate gradients
    total_grad_J2 = 0.0
    total_grad_J4 = 0.0
    total_grad_b = np.zeros(N)
    total_grad_J = np.zeros((N, N))

    count = 0
    for r in range(n_replicas):
        for k in range(n_steps):
            x_k = trajectories[r, k]
            x_kp1 = trajectories[r, k + 1]

            g_J2, g_J4, g_b, g_J = compute_gradients(
                x_k, x_kp1, params, mu, kT, dt
            )

            total_grad_J2 += g_J2
            total_grad_J4 += g_J4
            total_grad_b += g_b
            total_grad_J += g_J
            count += 1

    # Average
    total_grad_J2 /= count
    total_grad_J4 /= count
    total_grad_b /= count
    total_grad_J /= count

    # Enforce symmetry of J
    total_grad_J = 0.5 * (total_grad_J + total_grad_J.T)

    # Update parameters (gradient descent on negative log-likelihood)
    new_params = params.copy()
    new_params.J2 = params.J2 - lr * total_grad_J2
    new_params.J4 = params.J4 - lr * total_grad_J4
    new_params.b = params.b - lr * total_grad_b
    new_params.J = params.J - lr * total_grad_J

    # Keep J symmetric and zero diagonal
    new_params.J = 0.5 * (new_params.J + new_params.J.T)
    np.fill_diagonal(new_params.J, 0)

    return new_params


def param_distance(p1: Params, p2: Params) -> float:
    """Compute distance between two parameter sets."""
    d_J2 = (p1.J2 - p2.J2)**2
    d_J4 = (p1.J4 - p2.J4)**2
    d_b = np.sum((p1.b - p2.b)**2)
    d_J = np.sum((p1.J - p2.J)**2)
    return np.sqrt(d_J2 + d_J4 + d_b + d_J)


def main():
    # System parameters
    N = 8              # Number of nodes
    mu = 1.0           # Mobility
    kT = 1.0           # Temperature
    dt = 0.01          # Time step
    n_steps = 50       # Steps per trajectory
    n_replicas = 20    # Number of parallel trajectories
    n_epochs = 200     # Training epochs
    lr = 0.1           # Learning rate

    np.random.seed(42)

    # Create target parameters (what we want to learn)
    target_params = Params(
        J2=0.5,
        J4=0.1,
        b=np.random.randn(N) * 0.3,
        J=np.zeros((N, N))
    )
    # Sparse random couplings
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand() < 0.3:  # 30% connectivity
                val = np.random.randn() * 0.2
                target_params.J[i, j] = val
                target_params.J[j, i] = val

    # Initialize learnable parameters (random)
    learned_params = Params(
        J2=np.random.randn() * 0.1,
        J4=np.random.randn() * 0.1,
        b=np.random.randn(N) * 0.1,
        J=np.random.randn(N, N) * 0.05
    )
    learned_params.J = 0.5 * (learned_params.J + learned_params.J.T)
    np.fill_diagonal(learned_params.J, 0)

    print(f"Initial distance to target: {param_distance(learned_params, target_params):.4f}")
    print(f"Target J2={target_params.J2:.3f}, J4={target_params.J4:.3f}")
    print(f"Initial J2={learned_params.J2:.3f}, J4={learned_params.J4:.3f}")
    print()

    # Training loop
    distances = []
    losses = []

    for epoch in range(n_epochs):
        # Generate trajectories from TARGET distribution
        # (This simulates having "data" from the true system)
        trajectories = np.zeros((n_replicas, n_steps + 1, N))
        for r in range(n_replicas):
            x0 = np.random.randn(N) * 0.5
            trajectories[r] = generate_trajectory(
                x0, target_params, mu, kT, dt, n_steps
            )

        # Compute loss (negative log-likelihood under learned params)
        total_loss = 0.0
        count = 0
        for r in range(n_replicas):
            for k in range(n_steps):
                x_k = trajectories[r, k]
                x_kp1 = trajectories[r, k + 1]
                dx = x_kp1 - x_k

                # Expected displacement under learned params
                grad_V = potential_gradient(x_k, learned_params)
                expected_dx = -mu * grad_V * dt

                # Residual (should be ~ N(0, 2kT μ dt))
                residual = dx - expected_dx
                variance = 2 * kT * mu * dt

                # Negative log-likelihood (Gaussian)
                nll = 0.5 * np.sum(residual**2) / variance + 0.5 * N * np.log(2 * np.pi * variance)
                total_loss += nll
                count += 1

        avg_loss = total_loss / count
        losses.append(avg_loss)

        # Update parameters
        learned_params = train_step(
            trajectories, learned_params, mu, kT, dt, lr
        )

        dist = param_distance(learned_params, target_params)
        distances.append(dist)

        if epoch % 20 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, dist={dist:.4f}, "
                  f"J2={learned_params.J2:.3f}, J4={learned_params.J4:.3f}")

    print()
    print(f"Final distance to target: {distances[-1]:.4f}")
    print(f"Target b: {target_params.b}")
    print(f"Learned b: {learned_params.b}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Negative Log-Likelihood')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(distances)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Distance to Target')
    axes[1].set_title('Parameter Recovery')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('verify_results.png', dpi=150)
    print(f"\nResults saved to verify_results.png")

    # Final verification: compare distributions
    print("\n--- Distribution Comparison ---")
    print("Generating samples from target and learned params...")

    n_samples = 1000
    burn_in = 100

    # Sample from target
    x = np.random.randn(N) * 0.5
    target_samples = []
    for _ in range(burn_in + n_samples):
        x, _ = langevin_step(x, target_params, mu, kT, dt)
        if _ >= burn_in:
            target_samples.append(x.copy())
    target_samples = np.array(target_samples)

    # Sample from learned
    x = np.random.randn(N) * 0.5
    learned_samples = []
    for _ in range(burn_in + n_samples):
        x, _ = langevin_step(x, learned_params, mu, kT, dt)
        if _ >= burn_in:
            learned_samples.append(x.copy())
    learned_samples = np.array(learned_samples)

    # Compare means and variances
    target_mean = np.mean(target_samples, axis=0)
    learned_mean = np.mean(learned_samples, axis=0)
    target_var = np.var(target_samples, axis=0)
    learned_var = np.var(learned_samples, axis=0)

    print(f"Mean difference (L2): {np.linalg.norm(target_mean - learned_mean):.4f}")
    print(f"Variance difference (L2): {np.linalg.norm(target_var - learned_var):.4f}")


if __name__ == "__main__":
    main()
