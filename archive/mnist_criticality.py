"""
MNIST learning using the Criticality Engine's actual ansatz from the paper.

The potential (Eq. 4):
    ∂_i V = 2*J2*x_i + 4*J4*x_i³ + b_i + Σ_j J_ij*x_j

This is φ⁴ (Higgs/double-well) + Ising couplings:
- J2 < 0, J4 > 0: each node has bistable potential (two minima at ±√(-J2/2J4))
- J_ij couplings: align neighboring nodes (ferromagnetic if J_ij > 0)

This creates a MULTI-MODAL energy landscape, not Gaussian!
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_mnist():
    from sklearn.datasets import fetch_openml
    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int32)
    return X, y


class CriticalityEngine:
    """
    Langevin dynamics with φ⁴ + Ising potential (the paper's Eq. 4).

    V(x) = J2 Σ x_i² + J4 Σ x_i⁴ + Σ b_i x_i + ½ Σ_{ij} J_ij x_i x_j

    ∂_i V = 2*J2*x_i + 4*J4*x_i³ + b_i + Σ_j J_ij*x_j

    With J2 < 0, J4 > 0: bistable nodes coupled via J_ij.
    """

    def __init__(self, n_dim, J2=-1.0, J4=0.5, kT=1.0):
        self.n = n_dim
        self.J2 = J2  # Negative for double-well
        self.J4 = J4  # Positive for stability
        self.kT = kT

        # Learnable parameters
        self.b = np.zeros(n_dim)  # Local biases

        # Coupling matrix J_ij (symmetric, zero diagonal)
        # Use low-rank for efficiency: J = W @ W.T
        self.rank = 100
        self.W = np.random.randn(n_dim, self.rank) * 0.01

    def grad_V(self, x):
        """
        ∂_i V = 2*J2*x_i + 4*J4*x_i³ + b_i + (J @ x)_i

        Handles batched x: (batch, n) -> (batch, n)
        """
        # Quartic terms (element-wise)
        grad = 2 * self.J2 * x + 4 * self.J4 * (x ** 3) + self.b

        # Coupling term: J @ x where J = W @ W.T
        if x.ndim == 1:
            Jx = self.W @ (self.W.T @ x)
        else:
            Jx = (x @ self.W) @ self.W.T
        grad = grad + Jx

        return grad

    def drift(self, x):
        """Drift = -∂V/∂x (Langevin dynamics)"""
        return -self.grad_V(x)

    def langevin_step(self, x, dt=0.01):
        """Euler-Maruyama: dx = -∇V dt + √(2kT dt) dW"""
        noise = np.random.randn(*x.shape)
        return x + self.drift(x) * dt + np.sqrt(2 * self.kT * dt) * noise

    def train_epoch(self, X_data, lr_b=0.01, lr_W=0.001, sigma=0.3, batch_size=100):
        """
        Denoising score matching with the φ⁴ potential.

        Target: score at noisy point should point toward clean data.
        """
        n_samples = len(X_data)
        perm = np.random.permutation(n_samples)
        total_loss = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = X_data[perm[start:end]]
            bs = len(batch)

            # Add noise
            eps = np.random.randn(bs, self.n)
            x_noisy = batch + sigma * eps

            # Target score: points from noisy toward clean
            # For denoising score matching: target = -eps/sigma
            target_score = -eps / sigma

            # Model score = -∇V / kT
            model_score = -self.grad_V(x_noisy) / self.kT

            # Residual and loss
            residual = model_score - target_score
            loss = np.mean(np.sum(residual ** 2, axis=1))
            total_loss += loss * bs

            # Gradients w.r.t. parameters
            # ∂(model_score)/∂b = -1/kT
            # ∂loss/∂b = 2 * mean(residual) * (-1/kT)
            grad_b = -2 * residual.mean(0) / self.kT

            # ∂(model_score)/∂W: involves ∂(W W^T x)/∂W
            Wx = x_noisy @ self.W  # (bs, rank)
            grad_W = -2 / self.kT * (residual.T @ Wx) / bs

            # Gradient clipping
            grad_b = np.clip(grad_b, -1.0, 1.0)
            grad_W = np.clip(grad_W, -0.1, 0.1)

            # Update
            self.b -= lr_b * grad_b
            self.W -= lr_W * grad_W

        return total_loss / n_samples

    def denoise(self, x_noisy, n_steps=100, dt=0.02, return_trajectory=False):
        """Denoise by following the drift (low noise)"""
        x = x_noisy.copy()
        trajectory = [x.copy()] if return_trajectory else None

        for _ in range(n_steps):
            x = x + self.drift(x) * dt + np.sqrt(2 * self.kT * dt * 0.01) * np.random.randn(*x.shape)
            if return_trajectory:
                trajectory.append(x.copy())

        if return_trajectory:
            return x, trajectory
        return x

    def generate(self, n_samples, n_steps=500, dt=0.02, anneal=False):
        """Generate by Langevin sampling from random init"""
        x = np.random.randn(n_samples, self.n) * 0.5

        if anneal:
            # Annealed Langevin: start hot, cool down
            T_start, T_end = 5.0, 0.1
            for step in range(n_steps):
                # Exponential cooling schedule
                T = T_start * (T_end / T_start) ** (step / n_steps)
                noise = np.random.randn(*x.shape)
                x = x + self.drift(x) * dt + np.sqrt(2 * T * dt) * noise
        else:
            for _ in range(n_steps):
                x = self.langevin_step(x, dt)
        return x


def main():
    X, y = load_mnist()

    size = 14
    n_train = 5000
    print(f"Using {size}x{size} = {size*size} pixels")

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

    # Scale to [-1, 1] range (better for φ⁴ potential with minima at ±1)
    X_scaled = 2 * X_train - 1  # [0,1] -> [-1,1]

    n_dim = size * size

    print(f"\nTraining Criticality Engine (φ⁴ + Ising, {n_dim} dims)...")
    print(f"  J2={-1.0} (double-well), J4={0.5} (stabilizing)")

    # J2 < 0 creates double-well with minima at x = ±√(-J2/2J4) = ±1
    engine = CriticalityEngine(n_dim, J2=-1.0, J4=0.5, kT=0.5)

    n_epochs = 100
    for epoch in range(n_epochs):
        loss = engine.train_epoch(X_scaled, lr_b=0.01, lr_W=0.005, sigma=0.3, batch_size=100)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: score_loss = {loss:.4f}")

    # === Score alignment test ===
    print("\n=== Score Direction Test ===")
    idx = np.where(y_train == 3)[0][0]
    x_data = X_scaled[idx]
    x_noisy = x_data + np.random.randn(n_dim) * 0.3

    score = -engine.grad_V(x_noisy) / engine.kT
    direction = x_data - x_noisy
    alignment = np.dot(score, direction) / (np.linalg.norm(score) * np.linalg.norm(direction) + 1e-8)
    print(f"  Score-to-data alignment: {alignment:.3f} (should be positive)")

    # === Denoising ===
    print("\n=== Denoising ===")
    test_indices = [np.where(y_train == i)[0][0] for i in range(10)]
    originals = X_scaled[test_indices]

    noisy = originals + np.random.randn(10, n_dim) * 0.3
    denoised = engine.denoise(noisy, n_steps=200, dt=0.02)

    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for i in range(10):
        # Convert back to [0,1] for display
        orig_img = ((originals[i] + 1) / 2).reshape(size, size)
        noisy_img = ((noisy[i] + 1) / 2).reshape(size, size)
        denoised_img = ((denoised[i] + 1) / 2).reshape(size, size)

        axes[0, i].imshow(np.clip(orig_img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(str(i))
        axes[0, i].axis('off')

        axes[1, i].imshow(np.clip(noisy_img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

        axes[2, i].imshow(np.clip(denoised_img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')

    plt.suptitle('Criticality Engine (φ⁴ + Ising): Denoising', fontsize=14)
    plt.tight_layout()
    plt.savefig('criticality_denoise.png', dpi=150)
    print("Saved criticality_denoise.png")

    # === Trajectory ===
    print("\n=== Relaxation Trajectory ===")
    digit_idx = 3
    x_start = originals[digit_idx:digit_idx+1] + np.random.randn(1, n_dim) * 0.3
    _, trajectory = engine.denoise(x_start, n_steps=200, dt=0.02, return_trajectory=True)

    steps_to_show = [0, 5, 10, 20, 50, 100, 150, 200]
    fig, axes = plt.subplots(1, len(steps_to_show) + 1, figsize=(15, 2))

    orig_img = ((originals[digit_idx] + 1) / 2).reshape(size, size)
    axes[0].imshow(np.clip(orig_img, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i, step in enumerate(steps_to_show):
        img = ((trajectory[step][0] + 1) / 2).reshape(size, size)
        axes[i+1].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[i+1].set_title(f't={step}')
        axes[i+1].axis('off')

    plt.suptitle('Criticality Engine: φ⁴ Relaxation Trajectory', fontsize=12)
    plt.tight_layout()
    plt.savefig('criticality_trajectory.png', dpi=150)
    print("Saved criticality_trajectory.png")

    # === Generation (with annealing) ===
    print("\n=== Generation (Annealed) ===")
    generated = engine.generate(20, n_steps=1000, dt=0.02, anneal=True)

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(20):
        ax = axes[i // 10, i % 10]
        img = ((generated[i] + 1) / 2).reshape(size, size)
        ax.imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    plt.suptitle('Criticality Engine: Annealed Generation (T: 5→0.1)', fontsize=12)
    plt.tight_layout()
    plt.savefig('criticality_generated.png', dpi=150)
    print("Saved criticality_generated.png")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
