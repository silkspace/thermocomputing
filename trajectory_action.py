"""
The KEY insight: Trajectory likelihood via Onsager-Machlup action.

S[x(t)] = ∫ (ẋ + ∇V)² dt / (4μkT)

When S → 0 for observed paths, exp(-S) → 1: we're on the constraint surface.

The forward-backward ENFORCED trajectories:
- Detailed balance: P[forward] = P[backward]
- This CONSTRAINS the parameter space
- Learning finds parameters where observed paths have S ≈ 0

This is the Lagrangian multiplier view:
- Minimize S (action)
- Subject to: trajectories match observed data
- λ (temperature) controls the constraint strength
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class ActionBasedEngine(torch.nn.Module):
    """
    Learn parameters by minimizing the Onsager-Machlup action
    over forward-backward trajectories.
    """

    def __init__(self, n_dim, J2=-1.0, J4=0.5, kT=0.5):
        super().__init__()
        self.n_dim = n_dim
        self.J2 = J2
        self.J4 = J4
        self.kT = kT
        self.mu = 1.0

        # Learnable bias
        self.b = torch.nn.Parameter(torch.zeros(n_dim))

    def grad_V(self, x):
        """∂V/∂x"""
        return 2 * self.J2 * x + 4 * self.J4 * (x ** 3) + self.b

    def compute_action(self, trajectory, dt):
        """
        Compute Onsager-Machlup action for a trajectory.

        S = Σ_k (Δx_k + μ∇V·Δt)² / (4μkT·Δt)

        When parameters are correct, S → thermal noise level.
        """
        n_steps = len(trajectory) - 1
        action = 0.0

        for k in range(n_steps):
            x_k = trajectory[k]
            x_next = trajectory[k + 1]
            dx = x_next - x_k

            # Predicted displacement from drift
            grad = self.grad_V(x_k)
            predicted_dx = -self.mu * grad * dt

            # Action contribution: (observed - predicted)²
            residual = dx - predicted_dx
            action_k = (residual ** 2).sum(dim=-1) / (4 * self.mu * self.kT * dt)
            action = action + action_k

        return action  # Shape: (batch,)

    def compute_forward_backward_action(self, trajectory, dt):
        """
        Compute action for BOTH forward and backward paths.

        Detailed balance: S_forward should equal S_backward
        when parameters are correct.
        """
        # Forward action
        S_forward = self.compute_action(trajectory, dt)

        # Backward: reverse the trajectory
        trajectory_back = trajectory.flip(dims=[0])
        S_backward = self.compute_action(trajectory_back, dt)

        return S_forward, S_backward

    @torch.no_grad()
    def generate_trajectory(self, x0, n_steps, dt):
        """Generate trajectory from initial condition."""
        trajectory = [x0]
        x = x0.clone()

        for _ in range(n_steps):
            grad = self.grad_V(x)
            noise = torch.randn_like(x) * np.sqrt(2 * self.kT * self.mu * dt)
            x = x - self.mu * grad * dt + noise
            trajectory.append(x.clone())

        return torch.stack(trajectory)  # (n_steps+1, batch, n_dim)


def train_with_trajectory_action(model, x_data, b_true, n_epochs=100, lr=0.1,
                                  n_steps=20, dt=0.05):
    """
    Train by minimizing action over trajectories starting from data.

    Key: we want S → 0 for the observed forward-backward paths.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'action_forward': [],
        'action_backward': [],
        'action_diff': [],  # |S_forward - S_backward| (should be small by detailed balance)
        'bias_cosine': [],
        'epoch': []
    }

    n_samples = len(x_data)
    batch_size = min(100, n_samples)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=x_data.device)
        x_shuf = x_data[perm]

        epoch_S_fwd = 0.0
        epoch_S_bwd = 0.0
        n_batches = n_samples // batch_size

        for b in range(n_batches):
            x_batch = x_shuf[b*batch_size:(b+1)*batch_size]

            optimizer.zero_grad()

            # Generate trajectory from data
            with torch.no_grad():
                # Use ground truth model to generate "observed" trajectory
                # (In real hardware, this would be measured)
                traj = model.generate_trajectory(x_batch, n_steps, dt)

            # Now compute action with LEARNABLE parameters
            # Detach trajectory but keep model parameters differentiable
            traj_detached = traj.detach()
            traj_detached.requires_grad_(False)

            # Recompute action with current parameters
            S_fwd, S_bwd = model.compute_forward_backward_action(traj_detached, dt)

            # Loss: minimize total action + enforce detailed balance
            # When parameters are correct: S_fwd ≈ S_bwd ≈ thermal noise level
            loss = S_fwd.mean() + S_bwd.mean() + (S_fwd - S_bwd).abs().mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_S_fwd += S_fwd.mean().item()
            epoch_S_bwd += S_bwd.mean().item()

        epoch_S_fwd /= n_batches
        epoch_S_bwd /= n_batches

        # Track metrics
        history['action_forward'].append(epoch_S_fwd)
        history['action_backward'].append(epoch_S_bwd)
        history['action_diff'].append(abs(epoch_S_fwd - epoch_S_bwd))
        history['epoch'].append(epoch)

        with torch.no_grad():
            cos_sim = torch.nn.functional.cosine_similarity(
                model.b.unsqueeze(0), b_true.unsqueeze(0)
            ).item()
            history['bias_cosine'].append(cos_sim)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: S_fwd={epoch_S_fwd:.2f}, S_bwd={epoch_S_bwd:.2f}, "
                  f"|ΔS|={abs(epoch_S_fwd - epoch_S_bwd):.4f}, cos(b,b*)={cos_sim:.4f}")

    return history


def main():
    print(f"Device: {DEVICE}")
    print("\n" + "="*60)
    print("TRAJECTORY ACTION LEARNING")
    print("Key: S → 0 when parameters match observed dynamics")
    print("="*60)

    # Setup
    n_dim = 30
    n_samples = 1000

    # Ground truth
    torch.manual_seed(42)
    b_true = torch.randn(n_dim, device=DEVICE) * 2.0

    # Create ground truth model for generating "observed" trajectories
    gt_model = ActionBasedEngine(n_dim).to(DEVICE)
    with torch.no_grad():
        gt_model.b.copy_(b_true)

    # Generate equilibrium samples as starting points
    print("\nGenerating equilibrium samples from ground truth...")
    x_init = torch.randn(n_samples, n_dim, device=DEVICE)
    for _ in range(500):  # Burn-in
        grad = gt_model.grad_V(x_init)
        noise = torch.randn_like(x_init) * np.sqrt(2 * gt_model.kT * 0.02)
        x_init = x_init - gt_model.mu * grad * 0.02 + noise
    x_data = x_init.detach()

    # Train new model to recover b*
    print("\n=== Training via Trajectory Action ===")
    model = ActionBasedEngine(n_dim).to(DEVICE)

    history = train_with_trajectory_action(
        model, x_data, b_true,
        n_epochs=100,
        lr=0.1,
        n_steps=20,
        dt=0.05
    )

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Actions over training
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['action_forward'], label='S_forward')
    ax.plot(history['epoch'], history['action_backward'], label='S_backward')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Action S')
    ax.set_title('Onsager-Machlup Action (should converge)')
    ax.legend()

    # Action difference (detailed balance check)
    ax = axes[0, 1]
    ax.plot(history['epoch'], history['action_diff'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('|S_forward - S_backward|')
    ax.set_title('Detailed Balance Check (should → 0)')
    ax.set_yscale('log')

    # Parameter recovery
    ax = axes[1, 0]
    ax.plot(history['epoch'], history['bias_cosine'])
    ax.axhline(y=1, color='g', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('cos(b, b*)')
    ax.set_title('Alignment with True Parameters')
    ax.set_ylim([0, 1.1])

    # Scatter: learned vs true
    ax = axes[1, 1]
    with torch.no_grad():
        b_learned = model.b.cpu().numpy()
        b_truth = b_true.cpu().numpy()
    ax.scatter(b_truth, b_learned, alpha=0.6)
    lims = [min(b_truth.min(), b_learned.min()), max(b_truth.max(), b_learned.max())]
    ax.plot(lims, lims, 'r--', label='Perfect')
    ax.set_xlabel('True bias b*')
    ax.set_ylabel('Learned bias b')
    ax.set_title('Parameter Recovery')
    ax.legend()

    plt.suptitle('Learning via Trajectory Action: S → 0 on Constraint Surface', fontsize=14)
    plt.tight_layout()
    plt.savefig('trajectory_action_learning.png', dpi=150, bbox_inches='tight')
    print("\nSaved trajectory_action_learning.png")

    print(f"\n=== RESULTS ===")
    print(f"Final S_forward: {history['action_forward'][-1]:.2f}")
    print(f"Final S_backward: {history['action_backward'][-1]:.2f}")
    print(f"Final |ΔS|: {history['action_diff'][-1]:.4f}")
    print(f"Final cos(b, b*): {history['bias_cosine'][-1]:.4f}")


if __name__ == "__main__":
    main()
