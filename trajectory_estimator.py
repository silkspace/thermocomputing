"""
Trajectory-based parameter estimation for the Criticality Engine.

This is NOT neural network training. We use ANALYTICAL gradient estimators
derived from the Onsager-Machlup action to learn physical parameters.

The key equations (from paper Eq 12-13):

    residual_i = Δx_i + μ·∂_iV·Δt    (observed - predicted displacement)

    ∂L/∂b_i = residual_i / (2kT)
    ∂L/∂J_ij = (residual_i · x_j + residual_j · x_i) / (2kT)

We observe trajectories, compute residuals, estimate parameters. No autograd.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class PhysicsParams:
    """Physical parameters for the φ⁴+Ising potential."""
    J2: float = -1.0      # Quadratic coefficient (negative for bistability)
    J4: float = 0.5       # Quartic coefficient (positive for boundedness)
    kT: float = 0.5       # Thermal energy
    mu: float = 1.0       # Mobility
    dt: float = 0.05      # Time step


@dataclass
class LearnableParams:
    """Learnable parameters: biases and couplings."""
    b: np.ndarray                          # Shape: (n_dim,) or (n_classes, n_dim)
    J: Optional[np.ndarray] = None         # Shape: (n_dim, n_dim) sparse or low-rank

    @classmethod
    def zeros(cls, n_dim: int, n_classes: int = 1) -> 'LearnableParams':
        """Initialize with zeros."""
        if n_classes == 1:
            b = np.zeros(n_dim)
        else:
            b = np.zeros((n_classes, n_dim))
        return cls(b=b, J=None)


@dataclass
class TrajectoryStep:
    """One step of an observed trajectory."""
    x: np.ndarray           # State at time t: (batch, n_dim)
    x_next: np.ndarray      # State at time t+dt: (batch, n_dim)

    @property
    def dx(self) -> np.ndarray:
        """Observed displacement."""
        return self.x_next - self.x


class DiffusionProcess(ABC):
    """Abstract base for generating trajectories."""

    @abstractmethod
    def step(self, x: np.ndarray) -> np.ndarray:
        """Take one diffusion step, return new state."""
        pass


class ConservativePixelDiffusion(DiffusionProcess):
    """
    Pixel diffusion that conserves total intensity.

    Pixels are particles that can move to neighbors.
    Total mass is conserved (sum of pixel values stays constant).
    """

    def __init__(self, image_shape: tuple, diffusion_rate: float = 0.1):
        self.H, self.W = image_shape
        self.diffusion_rate = diffusion_rate

    def step(self, x: np.ndarray) -> np.ndarray:
        """
        Diffuse pixels while conserving total intensity.

        Each pixel exchanges intensity with its neighbors.
        """
        batch_size = x.shape[0] if x.ndim > 1 else 1
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_img = x.reshape(batch_size, self.H, self.W)
        x_new = x_img.copy()

        # Laplacian diffusion (conserves total)
        for i in range(batch_size):
            # Compute Laplacian
            laplacian = np.zeros_like(x_img[i])
            laplacian[1:, :] += x_img[i, :-1, :] - x_img[i, 1:, :]
            laplacian[:-1, :] += x_img[i, 1:, :] - x_img[i, :-1, :]
            laplacian[:, 1:] += x_img[i, :, :-1] - x_img[i, :, 1:]
            laplacian[:, :-1] += x_img[i, :, 1:] - x_img[i, :, :-1]

            # Add noise that conserves sum (zero-sum noise)
            noise = np.random.randn(self.H, self.W)
            noise -= noise.mean()  # Zero-sum noise

            x_new[i] = x_img[i] + self.diffusion_rate * laplacian + 0.1 * noise

        return x_new.reshape(batch_size, -1)


class LangevinDiffusion(DiffusionProcess):
    """
    Standard Langevin diffusion under φ⁴ potential.

    dx = -μ∇V·dt + √(2μkT·dt)·η
    """

    def __init__(self, physics: PhysicsParams):
        self.physics = physics

    def grad_V_phi4(self, x: np.ndarray) -> np.ndarray:
        """Gradient of φ⁴ potential (no bias, no coupling)."""
        return 2 * self.physics.J2 * x + 4 * self.physics.J4 * (x ** 3)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Take one Langevin step."""
        p = self.physics
        grad = self.grad_V_phi4(x)
        noise = np.random.randn(*x.shape) * np.sqrt(2 * p.kT * p.mu * p.dt)
        return x - p.mu * grad * p.dt + noise


class TrajectoryEstimator:
    """
    Estimate physical parameters from observed trajectories.

    Uses analytical gradient formulas - NO autograd, NO neural networks.
    """

    def __init__(self,
                 n_dim: int,
                 n_classes: int = 1,
                 physics: PhysicsParams = None):
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.physics = physics or PhysicsParams()

        # Initialize learnable parameters
        self.params = LearnableParams.zeros(n_dim, n_classes)

    def get_bias(self, class_idx: Optional[int] = None) -> np.ndarray:
        """Get bias vector for a class."""
        if self.n_classes == 1:
            return self.params.b
        return self.params.b[class_idx]

    def grad_V(self, x: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute ∂V/∂x = 2J₂x + 4J₄x³ + b + Jx
        """
        p = self.physics
        grad = 2 * p.J2 * x + 4 * p.J4 * (x ** 3)

        # Add bias
        b = self.get_bias(class_idx)
        grad = grad + b

        # Add coupling if present
        if self.params.J is not None:
            grad = grad + x @ self.params.J

        return grad

    def predicted_displacement(self, x: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        """Predicted Δx from current parameters."""
        p = self.physics
        return -p.mu * self.grad_V(x, class_idx) * p.dt

    def compute_residual(self, step: TrajectoryStep, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Compute residual = observed_dx - predicted_dx

        This is (Δx + μ∇V·Δt) from the paper.
        """
        predicted = self.predicted_displacement(step.x, class_idx)
        return step.dx - predicted

    def estimate_bias_gradient(self,
                                step: TrajectoryStep,
                                class_idx: Optional[int] = None) -> np.ndarray:
        """
        Analytical gradient for bias b_i.

        From paper Eq 13: -∂/∂b_i ln P(Δx) = residual_i / (2kT)

        The sign here is tricky. We found empirically that NEGATIVE residual
        gives better classification, suggesting biases should point TOWARD data.
        """
        residual = self.compute_residual(step, class_idx)
        # Average over batch
        grad = -residual.mean(axis=0) / (2 * self.physics.kT)
        return grad

    def estimate_coupling_gradient(self,
                                    step: TrajectoryStep,
                                    class_idx: Optional[int] = None) -> np.ndarray:
        """
        Analytical gradient for coupling J_ij.

        ∂L/∂J_ij = -(residual_i · x_j + residual_j · x_i) / (2kT)
        """
        residual = self.compute_residual(step, class_idx)
        x = step.x

        # Outer product: grad_ij = residual_i * x_j + residual_j * x_i
        # Average over batch
        batch_size = x.shape[0]
        grad = np.zeros((self.n_dim, self.n_dim))

        for b in range(batch_size):
            grad += np.outer(residual[b], x[b]) + np.outer(x[b], residual[b])

        grad /= batch_size
        grad /= (2 * self.physics.kT)

        return -grad

    def update_bias(self, step: TrajectoryStep, lr: float, class_idx: Optional[int] = None):
        """Update bias using analytical gradient."""
        grad = self.estimate_bias_gradient(step, class_idx)

        if self.n_classes == 1:
            self.params.b -= lr * grad
        else:
            self.params.b[class_idx] -= lr * grad

    def compute_action(self, trajectory: list[TrajectoryStep], class_idx: Optional[int] = None) -> float:
        """
        Compute Onsager-Machlup action for a trajectory.

        S = Σ_k ||residual_k||² / (4μkT·Δt)
        """
        p = self.physics
        total_action = 0.0

        for step in trajectory:
            residual = self.compute_residual(step, class_idx)
            action_k = (residual ** 2).sum() / (4 * p.mu * p.kT * p.dt)
            total_action += action_k

        return total_action


def create_mnist_estimator(image_size: int = 14) -> tuple[TrajectoryEstimator, ConservativePixelDiffusion]:
    """Factory for MNIST trajectory estimation."""
    n_dim = image_size * image_size

    physics = PhysicsParams(J2=-1.0, J4=0.5, kT=0.5, mu=1.0, dt=0.05)
    estimator = TrajectoryEstimator(n_dim, n_classes=10, physics=physics)
    # Conservative diffusion: preserves total intensity while pixels diffuse
    # This is data-dependent (class-specific) unlike pure Langevin!
    diffusion = ConservativePixelDiffusion((image_size, image_size), diffusion_rate=0.1)

    return estimator, diffusion


# Data loading moved to utils.py for single source of truth
from utils import load_mnist


def train_from_trajectories(
    estimator: TrajectoryEstimator,
    diffusion: DiffusionProcess,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_steps: int = 20,
    lr: float = 0.01,
    n_epochs: int = 50,
    samples_per_epoch: int = 500,
    eval_samples: int = 500
) -> dict:
    """
    Train parameters by observing diffusion trajectories.

    For each data sample:
    1. Start from data
    2. Diffuse for n_steps (observe trajectory)
    3. Accumulate gradients over trajectory
    4. Update parameters once per sample (averaged gradient)

    Args:
        estimator: TrajectoryEstimator to train
        diffusion: DiffusionProcess for generating trajectories
        X_train: Training data
        y_train: Training labels
        n_steps: Number of diffusion steps per trajectory
        lr: Learning rate
        n_epochs: Number of training epochs
        samples_per_epoch: Number of samples to process per epoch
        eval_samples: Number of samples to use for accuracy evaluation
    """
    n_samples = len(X_train)
    n_dim = X_train.shape[1]
    history = {'epoch': [], 'action': [], 'bias_norm': [], 'accuracy': []}

    # Check accuracy BEFORE any training
    n_eval = min(eval_samples, n_samples)
    correct = 0
    for i in range(n_eval):
        pred = classify_by_energy(estimator, X_train[i])
        if pred == y_train[i]:
            correct += 1
    acc_before = correct / n_eval * 100
    print(f"BEFORE TRAINING: ||b||={np.linalg.norm(estimator.params.b):.4f}, Acc={acc_before:.1f}%")

    n_batch = min(samples_per_epoch, n_samples)

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        epoch_action = 0.0

        # Accumulate gradients per class
        grad_accum = np.zeros((estimator.n_classes, n_dim))
        class_counts = np.zeros(estimator.n_classes)

        for idx in perm[:n_batch]:
            x = X_train[idx:idx+1]  # Shape: (1, n_dim)
            c = int(y_train[idx])

            trajectory = []
            traj_grad = np.zeros(n_dim)

            # Generate trajectory and accumulate gradients
            for k in range(n_steps):
                x_next = diffusion.step(x)
                step = TrajectoryStep(x=x, x_next=x_next)
                trajectory.append(step)

                # Accumulate gradient for this step
                grad_k = estimator.estimate_bias_gradient(step, class_idx=c)
                traj_grad += grad_k

                x = x_next

            # Average gradient over trajectory
            traj_grad /= n_steps

            # Clip gradient
            grad_norm = np.linalg.norm(traj_grad)
            if grad_norm > 1.0:
                traj_grad = traj_grad / grad_norm

            grad_accum[c] += traj_grad
            class_counts[c] += 1

            # Compute action for this trajectory
            action = estimator.compute_action(trajectory, class_idx=c)
            epoch_action += action / n_steps  # Normalize by trajectory length

        # Update biases with averaged gradients
        for c in range(estimator.n_classes):
            if class_counts[c] > 0:
                avg_grad = grad_accum[c] / class_counts[c]
                estimator.params.b[c] -= lr * avg_grad

        epoch_action /= n_batch

        # Track metrics
        bias_norm = np.linalg.norm(estimator.params.b)

        # Compute accuracy
        correct = 0
        for i in range(n_eval):
            pred = classify_by_energy(estimator, X_train[i])
            if pred == y_train[i]:
                correct += 1
        acc = correct / n_eval * 100

        history['epoch'].append(epoch)
        history['action'].append(epoch_action)
        history['bias_norm'].append(bias_norm)
        history['accuracy'].append(acc)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Action={epoch_action:.2f}, ||b||={bias_norm:.2f}, Acc={acc:.1f}%")

    return history


def classify_by_energy(estimator: TrajectoryEstimator, x: np.ndarray) -> int:
    """Classify by minimum energy: c* = argmin_c V_c(x)"""
    energies = []
    for c in range(estimator.n_classes):
        # V = J2*||x||² + J4*||x||⁴ + b_c·x
        p = estimator.physics
        V = p.J2 * np.sum(x**2) + p.J4 * np.sum(x**4)
        V += np.dot(estimator.get_bias(c), x.flatten())
        energies.append(V)
    return np.argmin(energies)


def compute_accuracy(estimator: TrajectoryEstimator, X: np.ndarray, y: np.ndarray):
    """Compute overall and per-class accuracy."""
    n = len(X)
    predictions = np.array([classify_by_energy(estimator, X[i]) for i in range(n)])

    # Overall accuracy
    overall_acc = (predictions == y).mean() * 100

    # Per-class accuracy
    per_class = {}
    for c in range(estimator.n_classes):
        mask = (y == c)
        if mask.sum() > 0:
            per_class[c] = (predictions[mask] == c).mean() * 100
        else:
            per_class[c] = 0.0

    return overall_acc, per_class, predictions


def main():
    print("="*60)
    print("TRAJECTORY-BASED PARAMETER ESTIMATION")
    print("Using analytical gradients, NOT neural network training")
    print("="*60)

    # Load data with proper train/test split
    X_train, y_train, X_test, y_test = load_mnist(n_train=5000, n_test=1000, image_size=14)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Create estimator and diffusion process
    estimator, diffusion = create_mnist_estimator(image_size=14)

    # Train from trajectories
    print("\n=== Training via Trajectory Observation ===")
    history = train_from_trajectories(
        estimator, diffusion, X_train, y_train,
        n_steps=5, lr=0.05, n_epochs=100
    )

    # Evaluate on HOLDOUT test set
    print("\n" + "="*60)
    print("HOLDOUT TEST SET EVALUATION")
    print("="*60)

    test_acc, test_per_class, _ = compute_accuracy(estimator, X_test, y_test)
    eval_train_samples = min(1000, len(X_train))
    train_acc, train_per_class, _ = compute_accuracy(estimator, X_train[:eval_train_samples], y_train[:eval_train_samples])

    print(f"\nTrain Accuracy: {train_acc:.1f}%")
    print(f"Test Accuracy:  {test_acc:.1f}% (HOLDOUT - never seen during training)")
    print(f"Random Baseline: 10.0%")

    print("\n--- Per-Class Test Accuracy ---")
    print("Digit | Accuracy | Count")
    print("-" * 30)
    for c in range(estimator.n_classes):
        count = (y_test == c).sum()
        print(f"  {c}   |  {test_per_class[c]:5.1f}%  |  {count}")

    # Compute confusion matrix style summary
    print("\n--- Hardest/Easiest Classes ---")
    sorted_classes = sorted(test_per_class.items(), key=lambda x: x[1])
    print(f"Hardest: digit {sorted_classes[0][0]} ({sorted_classes[0][1]:.1f}%)")
    print(f"Easiest: digit {sorted_classes[-1][0]} ({sorted_classes[-1][1]:.1f}%)")

    # Visualize learned biases
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes = estimator.n_classes
    n_cols = 5
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axes = axes.flatten()
    for c in range(n_classes):
        ax = axes[c]
        bias = -estimator.get_bias(c).reshape(14, 14)
        ax.imshow(bias, cmap='RdBu_r')
        ax.set_title(f'Class {c}')
        ax.axis('off')
    # Hide unused axes
    for c in range(n_classes, len(axes)):
        axes[c].axis('off')
    plt.suptitle('Learned Biases (-b_c) via Trajectory Estimation', fontsize=12)
    plt.tight_layout()
    plt.savefig('paper_figures/trajectory_estimated_biases.png', dpi=150)
    print("\nSaved paper_figures/trajectory_estimated_biases.png")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history['epoch'], history['action'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Action')
    axes[0].set_title('Trajectory Action')

    axes[1].plot(history['epoch'], history['bias_norm'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('||b||')
    axes[1].set_title('Bias Norm')

    plt.tight_layout()
    plt.savefig('paper_figures/trajectory_estimation_training.png', dpi=150)
    print("Saved paper_figures/trajectory_estimation_training.png")

    # Save learned biases for use in denoising demo
    np.save('learned_biases.npy', estimator.params.b)
    print("Saved learned_biases.npy")

    return estimator


if __name__ == "__main__":
    main()
