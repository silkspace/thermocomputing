"""
PyTorch implementation of the Criticality Engine.

Core equations from the paper:
- Eq 6: ∂_i V = 2J₂x_i + 4J₄x_i³ + b_i + Σⱼ J_ij x_j
- Eq 7-8: Gradient of log-likelihood = (predicted_force - observed_velocity) × neighbor_state
- Eq 10: Velocity matching interpretation

This demonstrates the key insight: learning converges when predicted force matches observed velocity.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Check for MPS (Apple Silicon) or CUDA
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


class CriticalityEngine(nn.Module):
    """
    φ⁴ + Ising potential with learnable parameters.

    V(x) = J₂ Σᵢ xᵢ² + J₄ Σᵢ xᵢ⁴ + Σᵢ bᵢxᵢ + ½ Σᵢⱼ Jᵢⱼ xᵢxⱼ

    For class-conditional: each class k has its own bias b_k
    """

    def __init__(self, n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=0.5, rank=32):
        super().__init__()
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.J2 = J2
        self.J4 = J4
        self.kT = kT
        self.mu = 1.0  # mobility

        # Learnable parameters
        # Class-conditional biases: b[k] for class k
        self.b = nn.Parameter(torch.zeros(n_classes, n_dim))

        # Low-rank coupling: J = W @ W.T (positive semi-definite)
        self.W = nn.Parameter(torch.randn(n_dim, rank) * 0.01)

    def grad_V(self, x, class_idx=None):
        """
        Compute ∂V/∂x = 2J₂x + 4J₄x³ + b + Jx

        Args:
            x: (batch, n_dim) state vectors
            class_idx: int or (batch,) tensor of class indices
        """
        # φ⁴ terms (per-pixel bistability)
        grad = 2 * self.J2 * x + 4 * self.J4 * (x ** 3)

        # Class-conditional bias
        if class_idx is not None:
            if isinstance(class_idx, int):
                grad = grad + self.b[class_idx]
            else:
                # Batched: each sample may have different class
                grad = grad + self.b[class_idx]

        # Coupling term: Jx where J = W @ W.T
        Jx = x @ self.W @ self.W.T
        grad = grad + Jx

        return grad

    def energy(self, x, class_idx=None):
        """Compute V(x)"""
        # φ⁴ terms
        V = self.J2 * (x ** 2).sum(dim=-1) + self.J4 * (x ** 4).sum(dim=-1)

        # Bias term
        if class_idx is not None:
            if isinstance(class_idx, int):
                V = V + (self.b[class_idx] * x).sum(dim=-1)
            else:
                # Select bias for each sample's class
                b_selected = self.b[class_idx]  # (batch, n_dim)
                V = V + (b_selected * x).sum(dim=-1)

        # Coupling term: ½ x.T @ J @ x
        Wx = x @ self.W  # (batch, rank)
        V = V + 0.5 * (Wx ** 2).sum(dim=-1)

        return V

    def langevin_step(self, x, class_idx, dt=0.01):
        """
        Single Langevin step: dx = -μ∇V dt + √(2kTμ dt) η

        Returns:
            x_new: updated state
            dx: displacement (observed velocity × dt)
        """
        grad = self.grad_V(x, class_idx)
        noise = torch.randn_like(x) * np.sqrt(2 * self.kT * self.mu * dt)
        dx = -self.mu * grad * dt + noise
        x_new = x + dx
        return x_new, dx

    def velocity_matching_loss(self, x, dx, class_idx, dt):
        """
        Eq 7-8: Loss = ||predicted_force × dt - observed_dx||² / (4 kT dt)

        This is the negative log-likelihood of the observed step.
        """
        grad = self.grad_V(x, class_idx)
        predicted_dx = -self.mu * grad * dt

        # Gaussian log-likelihood: (dx - predicted_dx)² / (2 × variance)
        # variance = 2 kT μ dt
        residual = dx - predicted_dx
        loss = (residual ** 2).sum(dim=-1) / (4 * self.kT * self.mu * dt)

        return loss.mean()

    def denoising_score_loss(self, x_clean, x_noisy, class_idx, sigma):
        """
        Denoising score matching (matching numpy implementation).

        target_score = -noise/σ  (where noise = x_noisy - x_clean = σ*eps)
        model_score = -∇V/kT

        Loss = ||model_score - target_score||²
        """
        noise = x_noisy - x_clean
        eps = noise / sigma  # eps ~ N(0,1)

        # Target score (direction back to clean data)
        target_score = -eps / sigma  # = -noise/σ²

        # Model score: -∇V/kT
        grad_V = self.grad_V(x_noisy, class_idx)
        model_score = -grad_V / self.kT

        # MSE loss
        residual = model_score - target_score
        loss = (residual ** 2).sum(dim=-1)
        return loss.mean()


def train_with_convergence_tracking(engine, X_data, y_data, n_epochs=100,
                                     lr=0.01, sigma=0.3, use_denoising=True):
    """
    Train the engine and track convergence of biases.

    Args:
        use_denoising: If True, use denoising score matching (what works).
                      If False, use trajectory velocity matching (Eq 7-8).
    """
    optimizer = torch.optim.Adam(engine.parameters(), lr=lr)

    history = {
        'loss': [],
        'bias_norm': [],
        'accuracy': [],
        'epoch': []
    }

    n_samples = len(X_data)
    batch_size = 100

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(n_samples, device=X_data.device)
        X_shuf = X_data[perm]
        y_shuf = y_data[perm]

        epoch_loss = 0.0
        n_batches = n_samples // batch_size

        for b in range(n_batches):
            X_batch = X_shuf[b*batch_size:(b+1)*batch_size]
            y_batch = y_shuf[b*batch_size:(b+1)*batch_size]

            optimizer.zero_grad()

            if use_denoising:
                # Denoising score matching
                noise = sigma * torch.randn_like(X_batch)
                x_noisy = X_batch + noise
                loss = engine.denoising_score_loss(X_batch, x_noisy, y_batch, sigma)
            else:
                # Trajectory velocity matching (Eq 7-8)
                dt = 0.02
                x = X_batch + sigma * torch.randn_like(X_batch)
                x_new, dx = engine.langevin_step(x, y_batch, dt=dt)
                loss = engine.velocity_matching_loss(x, dx, y_batch, dt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= n_batches

        # Track metrics
        history['loss'].append(epoch_loss)
        history['epoch'].append(epoch)

        # Bias norms per class
        with torch.no_grad():
            bias_norms = torch.norm(engine.b, dim=1).cpu().numpy()
            history['bias_norm'].append(bias_norms.copy())

        # Classification accuracy
        if epoch % 10 == 0:
            acc = test_classification(engine, X_data[:1000], y_data[:1000])
            history['accuracy'].append((epoch, acc))
            print(f"Epoch {epoch}: loss={epoch_loss:.2f}, acc={acc:.1f}%")

    return history


def test_classification(engine, X, y):
    """Energy-based classification: k* = argmin_k V_k(x)"""
    with torch.no_grad():
        # Compute energy for each class
        energies = []
        for k in range(engine.n_classes):
            V_k = engine.energy(X, class_idx=k)
            energies.append(V_k)

        energies = torch.stack(energies, dim=1)  # (batch, n_classes)
        predictions = torch.argmin(energies, dim=1)

        correct = (predictions == y).sum().item()
        return 100.0 * correct / len(y)


def plot_convergence(history, save_path='convergence_pytorch.png'):
    """Plot learning convergence."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss over epochs
    ax = axes[0, 0]
    ax.plot(history['epoch'], history['loss'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score Matching Loss')
    ax.set_title('Training Loss')

    # Bias norms over epochs
    ax = axes[0, 1]
    bias_norms = np.array(history['bias_norm'])
    for k in range(10):
        ax.plot(history['epoch'], bias_norms[:, k], label=f'Class {k}', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('||b_k||')
    ax.set_title('Bias Magnitude per Class (Convergence)')
    ax.legend(fontsize=8, ncol=2)

    # Zoomed loss (first 20 epochs)
    ax = axes[1, 0]
    n_zoom = min(20, len(history['loss']))
    ax.plot(history['epoch'][:n_zoom], history['loss'][:n_zoom], 'o-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss (First 20 Epochs) - Learning Speed')

    # Classification accuracy
    ax = axes[1, 1]
    if history['accuracy']:
        epochs, accs = zip(*history['accuracy'])
        ax.plot(epochs, accs, 'o-', linewidth=2, markersize=8)
        ax.axhline(y=10, color='r', linestyle='--', label='Random baseline', alpha=0.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Classification Accuracy')
        ax.legend()
        ax.set_ylim([0, max(accs) * 1.2])

    plt.suptitle('Criticality Engine: Learning Convergence (PyTorch/MPS)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved {save_path}")


def load_mnist_torch():
    """Load MNIST and convert to PyTorch tensors."""
    from mnist_conditional import load_mnist
    X, y = load_mnist()

    size = 14
    n_train = 5000

    # Downsample
    X_train = X[:n_train].reshape(-1, 28, 28)
    factor = 28 // size
    X_down = np.zeros((n_train, size, size))
    for i in range(size):
        for j in range(size):
            X_down[:, i, j] = X_train[:,
                i*factor:(i+1)*factor,
                j*factor:(j+1)*factor
            ].mean(axis=(1, 2))
    X_train = X_down.reshape(n_train, -1)
    y_train = y[:n_train]

    # Scale to [-1, 1]
    X_scaled = 2 * X_train - 1

    # Convert to torch
    X_torch = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
    y_torch = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    return X_torch, y_torch, size


def main():
    print("Loading MNIST...")
    X, y, size = load_mnist_torch()
    n_dim = size * size

    print(f"Data shape: {X.shape}, Device: {X.device}")

    # Create engine
    engine = CriticalityEngine(
        n_dim=n_dim,
        n_classes=10,
        J2=-1.0,
        J4=0.5,
        kT=0.5,
        rank=32
    ).to(DEVICE)

    print(f"\nModel parameters:")
    print(f"  Biases b: {engine.b.shape}")
    print(f"  Couplings W: {engine.W.shape}")

    # Train with convergence tracking
    print("\n=== Training (Denoising Score Matching) ===")
    history = train_with_convergence_tracking(
        engine, X, y,
        n_epochs=100,
        lr=0.01,
        sigma=0.3,
        use_denoising=True
    )

    # Plot convergence
    plot_convergence(history)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_acc = test_classification(engine, X, y)
    print(f"Final classification accuracy: {final_acc:.1f}%")

    # Visualize learned biases
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    with torch.no_grad():
        biases = -engine.b.cpu().numpy()

    for k in range(10):
        ax = axes[k // 5, k % 5]
        bias_img = biases[k].reshape(size, size)
        ax.imshow(bias_img, cmap='RdBu_r',
                 vmin=-np.abs(bias_img).max(), vmax=np.abs(bias_img).max())
        ax.set_title(f'$-b_{k}$')
        ax.axis('off')

    plt.suptitle('Learned Class Biases (PyTorch)', fontsize=12)
    plt.tight_layout()
    plt.savefig('biases_pytorch.png', dpi=150, bbox_inches='tight')
    print("Saved biases_pytorch.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
