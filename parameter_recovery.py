"""
Parameter Recovery from Trajectory Observation

The paper's core claim:
- System has energy V(x) with parameters θ = {J, b}
- Thermal dynamics: dx = -μ∇V dt + √(2μkT) dW
- We OBSERVE trajectories at discrete times
- Apply gradient formulas (Eq 12-14) to ESTIMATE θ

This script:
1. Create a system with KNOWN J*, b* (ground truth)
2. Simulate Langevin dynamics
3. Record trajectories
4. Learn J, b using ONLY the recorded observations
5. Compare learned vs true parameters

NO tricks. NO assumptions. Just trajectory → parameters.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

Path('paper_figures').mkdir(exist_ok=True)


def create_ground_truth(n_dim: int, coupling_strength: float = 0.5,
                         bias_strength: float = 1.0, seed: int = 42):
    """Create ground truth parameters J*, b*. J must be positive definite for stable dynamics."""
    np.random.seed(seed)

    # Create positive definite J via J = A'A + λI
    A = np.random.randn(n_dim, n_dim) * coupling_strength / np.sqrt(n_dim)
    J_true = A.T @ A + 0.5 * np.eye(n_dim)  # Positive definite

    # Random bias
    b_true = np.random.randn(n_dim) * bias_strength

    print(f"J eigenvalues: min={np.linalg.eigvalsh(J_true).min():.3f}, max={np.linalg.eigvalsh(J_true).max():.3f}")

    return J_true, b_true


def compute_force(x: np.ndarray, J: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Force from potential V(x) = 0.5 x'Jx + b'x
    f = -∇V = -Jx - b
    """
    return -J @ x - b


def langevin_step(x: np.ndarray, J: np.ndarray, b: np.ndarray,
                   mu: float, kT: float, dt: float) -> tuple:
    """
    One step of overdamped Langevin dynamics.

    dx = -μ∇V dt + √(2μkT dt) η
       = μ(-Jx - b) dt + noise

    Returns new state and the displacement Δx.
    """
    force = compute_force(x, J, b)
    noise = np.sqrt(2 * mu * kT * dt) * np.random.randn(len(x))

    dx = mu * force * dt + noise
    x_new = x + dx

    return x_new, dx


def simulate_trajectory(x0: np.ndarray, J: np.ndarray, b: np.ndarray,
                        n_steps: int, mu: float = 1.0, kT: float = 1.0,
                        dt: float = 0.01) -> tuple:
    """
    Simulate a trajectory and record observations.

    Returns:
    - states: array of shape (n_steps+1, n_dim), the x(t_k)
    - displacements: array of shape (n_steps, n_dim), the Δx^k = x^{k+1} - x^k
    """
    n_dim = len(x0)
    states = np.zeros((n_steps + 1, n_dim))
    displacements = np.zeros((n_steps, n_dim))

    states[0] = x0.copy()
    x = x0.copy()

    for k in range(n_steps):
        x_new, dx = langevin_step(x, J, b, mu, kT, dt)
        states[k + 1] = x_new
        displacements[k] = dx
        x = x_new

    return states, displacements


def compute_J_gradient(x: np.ndarray, dx: np.ndarray, J: np.ndarray,
                        b: np.ndarray, mu: float, kT: float, dt: float) -> np.ndarray:
    """
    Gradient of log-likelihood w.r.t. J (Equation 12 from paper).

    The observed displacement is dx = Δx^k
    The predicted drift is: μ * f * dt = μ * (-Jx - b) * dt

    Residual: r = dx - predicted_drift = dx + μ(Jx + b)dt

    For coupling J_ij, the gradient is (from Eq 12):
    ∂(-log P)/∂J_ij = (r_i * x_j + r_j * x_i) / (2 kT)

    We want gradient ASCENT on log P, so return negative of above.
    """
    n_dim = len(x)

    # Predicted drift under current parameters
    predicted_drift = mu * (-J @ x - b) * dt

    # Residual: observed - predicted
    residual = dx - predicted_drift

    # Gradient (Eq 12): for each (i,j), it's residual_i * x_j + residual_j * x_i
    # This is the outer product plus its transpose, divided by 2
    grad = np.outer(residual, x) + np.outer(x, residual)
    grad = grad / (2 * kT)

    return grad


def compute_b_gradient(x: np.ndarray, dx: np.ndarray, J: np.ndarray,
                        b: np.ndarray, mu: float, kT: float, dt: float) -> np.ndarray:
    """
    Gradient of log-likelihood w.r.t. b (Equation 13/14 from paper).

    ∂(-log P)/∂b_i = r_i / kT

    where r_i = dx_i - μ(-J_i·x - b_i)dt is the residual.
    """
    predicted_drift = mu * (-J @ x - b) * dt
    residual = dx - predicted_drift

    grad = residual / kT

    return grad


def learn_from_trajectory(states: np.ndarray, displacements: np.ndarray,
                           mu: float, kT: float, dt: float,
                           lr_J: float = 0.1, lr_b: float = 0.01,
                           n_epochs: int = 20) -> tuple:
    """
    Learn J, b from observed trajectory using gradient descent.

    This is the core algorithm:
    - For each timestep k, we have x^k and Δx^k
    - Compute gradients using Eq 12-14
    - Update parameters
    """
    n_steps, n_dim = displacements.shape

    # Initialize J to small identity (not zero - breaks degeneracy)
    J_learned = 0.1 * np.eye(n_dim)
    b_learned = np.zeros(n_dim)

    history = {'J_error': [], 'b_error': [], 'J_norm': [], 'b_norm': [], 'residual': []}

    for epoch in range(n_epochs):
        # Accumulate gradients over trajectory
        grad_J_total = np.zeros((n_dim, n_dim))
        grad_b_total = np.zeros(n_dim)

        for k in range(n_steps):
            x_k = states[k]
            dx_k = displacements[k]

            grad_J = compute_J_gradient(x_k, dx_k, J_learned, b_learned, mu, kT, dt)
            grad_b = compute_b_gradient(x_k, dx_k, J_learned, b_learned, mu, kT, dt)

            grad_J_total += grad_J
            grad_b_total += grad_b

        # Average over timesteps
        grad_J_total /= n_steps
        grad_b_total /= n_steps

        # Clip gradients
        grad_J_norm = np.linalg.norm(grad_J_total)
        grad_b_norm = np.linalg.norm(grad_b_total)
        if grad_J_norm > 10:
            grad_J_total = grad_J_total * 10 / grad_J_norm
        if grad_b_norm > 10:
            grad_b_total = grad_b_total * 10 / grad_b_norm

        # Gradient DESCENT on negative log-likelihood = ASCENT on log-likelihood
        # But our gradients are ∂(-log P)/∂θ, so we subtract
        J_learned -= lr_J * grad_J_total
        b_learned -= lr_b * grad_b_total

        # Enforce symmetry on J
        J_learned = (J_learned + J_learned.T) / 2

        # Track mean residual
        total_residual = 0
        for k in range(n_steps):
            pred = mu * (-J_learned @ states[k] - b_learned) * dt
            res = displacements[k] - pred
            total_residual += np.linalg.norm(res)
        mean_residual = total_residual / n_steps

        history['J_norm'].append(np.linalg.norm(J_learned))
        history['b_norm'].append(np.linalg.norm(b_learned))
        history['residual'].append(mean_residual)

        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: ||J||={np.linalg.norm(J_learned):.4f}, ||b||={np.linalg.norm(b_learned):.4f}, <res>={mean_residual:.6f}")

    return J_learned, b_learned, history


def parameter_recovery_experiment():
    """
    Full parameter recovery experiment.

    1. Create ground truth J*, b*
    2. Simulate MULTIPLE trajectories from different starting points
    3. Learn J, b from observations
    4. Compare
    """
    print("=" * 60)
    print("PARAMETER RECOVERY FROM TRAJECTORY OBSERVATION")
    print("=" * 60)

    # Setup - lower temperature for better SNR
    n_dim = 10
    mu = 1.0
    kT = 0.1  # Lower temperature = less noise = more signal
    dt = 0.01
    n_steps_per_traj = 1000
    n_trajectories = 10  # Multiple trajectories from different starting points

    print(f"\nSystem: n_dim={n_dim}, μ={mu}, kT={kT}, dt={dt}")
    print(f"Trajectories: {n_trajectories} x {n_steps_per_traj} steps")

    # Create ground truth
    J_true, b_true = create_ground_truth(n_dim, coupling_strength=0.3, bias_strength=0.5)
    print(f"\nGround truth: ||J*||={np.linalg.norm(J_true):.4f}, ||b*||={np.linalg.norm(b_true):.4f}")

    # Simulate MULTIPLE trajectories from diverse starting points
    print("\nSimulating trajectories...")
    all_states = []
    all_displacements = []

    for traj_idx in range(n_trajectories):
        # Start from different points (some far from equilibrium)
        x0 = np.random.randn(n_dim) * 3.0  # Diverse starting points

        states, displacements = simulate_trajectory(x0, J_true, b_true, n_steps_per_traj, mu, kT, dt)
        all_states.append(states[:-1])  # Exclude last state (no displacement)
        all_displacements.append(displacements)

    # Concatenate all trajectories
    states = np.vstack(all_states)
    displacements = np.vstack(all_displacements)
    print(f"Total observations: {states.shape[0]} state-displacement pairs")

    # Learn from trajectory
    print("\nLearning from observations...")
    J_learned, b_learned, history = learn_from_trajectory(
        states, displacements, mu, kT, dt,
        lr_J=0.1, lr_b=1.0, n_epochs=1000  # Higher LR, more epochs for convergence
    )

    # Compare
    J_error = np.linalg.norm(J_learned - J_true) / np.linalg.norm(J_true)
    b_error = np.linalg.norm(b_learned - b_true) / np.linalg.norm(b_true)

    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"||J_learned||  = {np.linalg.norm(J_learned):.4f}")
    print(f"||J_true||     = {np.linalg.norm(J_true):.4f}")
    print(f"J relative error: {J_error*100:.2f}%")
    print()
    print(f"||b_learned||  = {np.linalg.norm(b_learned):.4f}")
    print(f"||b_true||     = {np.linalg.norm(b_true):.4f}")
    print(f"b relative error: {b_error*100:.2f}%")
    print()

    J_corr = np.corrcoef(J_learned.flatten(), J_true.flatten())[0, 1]
    b_corr = np.corrcoef(b_learned.flatten(), b_true.flatten())[0, 1]
    print(f"J correlation: {J_corr:.4f}")
    print(f"b correlation: {b_corr:.4f}")

    # Recovery success?
    if J_error < 0.1 and b_error < 0.1:
        print("\n✓ SUCCESS: Parameters recovered with <10% error!")
    elif J_corr > 0.95 and b_corr > 0.95:
        print("\n✓ SUCCESS: Parameters highly correlated (>0.95)!")
    else:
        print("\n⚠ Parameters partially recovered")

    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # J matrices
    vmax = max(np.abs(J_true).max(), np.abs(J_learned).max())
    axes[0, 0].imshow(J_true, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title('J_true')
    axes[0, 1].imshow(J_learned, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[0, 1].set_title('J_learned')
    axes[0, 2].scatter(J_true.flatten(), J_learned.flatten(), alpha=0.5, s=10)
    axes[0, 2].plot([-vmax, vmax], [-vmax, vmax], 'k--', alpha=0.5)
    axes[0, 2].set_xlabel('J_true')
    axes[0, 2].set_ylabel('J_learned')
    axes[0, 2].set_title(f'J correlation: {J_corr:.3f}')

    # b vectors
    axes[1, 0].bar(range(n_dim), b_true, alpha=0.7, label='true')
    axes[1, 0].bar(range(n_dim), b_learned, alpha=0.5, label='learned')
    axes[1, 0].legend()
    axes[1, 0].set_title('Bias vectors')

    axes[1, 1].scatter(b_true, b_learned, alpha=0.7)
    bmax = max(np.abs(b_true).max(), np.abs(b_learned).max())
    axes[1, 1].plot([-bmax, bmax], [-bmax, bmax], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('b_true')
    axes[1, 1].set_ylabel('b_learned')
    axes[1, 1].set_title(f'b correlation: {b_corr:.3f}')

    # Learning curves
    axes[1, 2].plot(history['J_norm'], label='||J||')
    axes[1, 2].plot(history['b_norm'], label='||b||')
    axes[1, 2].axhline(np.linalg.norm(J_true), color='blue', linestyle='--', alpha=0.5)
    axes[1, 2].axhline(np.linalg.norm(b_true), color='orange', linestyle='--', alpha=0.5)
    axes[1, 2].legend()
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_title('Parameter norms during learning')

    plt.tight_layout()
    plt.savefig('paper_figures/parameter_recovery.png', dpi=150)
    print("\nSaved paper_figures/parameter_recovery.png")
    plt.close()

    return J_true, J_learned, b_true, b_learned


if __name__ == "__main__":
    parameter_recovery_experiment()
