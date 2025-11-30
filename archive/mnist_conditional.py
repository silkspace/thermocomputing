"""
Class-conditional Criticality Engine.

Key idea: Instead of one potential for all digits, learn per-class modulation.
Each digit k has its own bias vector b_k, creating digit-specific attractors.

Like NMF: M = WH where H encodes "which digit"
Like Hopfield: stored patterns as attractors
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


class ConditionalCriticalityEngine:
    """
    φ⁴ + Ising with per-class bias modulation.

    V(x; k) = J2 Σx² + J4 Σx⁴ + b_k·x + ½x^T J x

    Each class k has its own bias b_k, creating class-specific attractors.
    """

    def __init__(self, n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=1.0):
        self.n = n_dim
        self.n_classes = n_classes
        self.J2 = J2
        self.J4 = J4
        self.kT = kT

        # Per-class biases (the key innovation)
        self.b = np.zeros((n_classes, n_dim))

        # Shared coupling matrix (low-rank)
        self.rank = 100
        self.W = np.random.randn(n_dim, self.rank) * 0.01

    def grad_V(self, x, class_idx=None):
        """
        ∂_i V = 2J₂x_i + 4J₄x_i³ + b_k,i + (Jx)_i

        If class_idx is None, use mean bias (for unconditional).
        """
        # φ⁴ terms
        grad = 2 * self.J2 * x + 4 * self.J4 * (x ** 3)

        # Class-specific bias
        if class_idx is not None:
            if isinstance(class_idx, (int, np.integer)):
                grad = grad + self.b[class_idx]
            else:
                # Batched: class_idx is array of labels
                grad = grad + self.b[class_idx]
        else:
            # Unconditional: use mean bias
            grad = grad + self.b.mean(0)

        # Coupling term
        if x.ndim == 1:
            Jx = self.W @ (self.W.T @ x)
        else:
            Jx = (x @ self.W) @ self.W.T
        grad = grad + Jx

        return grad

    def drift(self, x, class_idx=None):
        return -self.grad_V(x, class_idx)

    def train_epoch(self, X_data, y_data, lr_b=0.01, lr_W=0.005, sigma=0.3, batch_size=100):
        """
        Denoising score matching with class-conditional biases.
        """
        n_samples = len(X_data)
        perm = np.random.permutation(n_samples)
        total_loss = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = perm[start:end]
            batch = X_data[idx]
            labels = y_data[idx]
            bs = len(batch)

            # Add noise
            eps = np.random.randn(bs, self.n)
            x_noisy = batch + sigma * eps

            # Target score
            target_score = -eps / sigma

            # Model score with class-specific bias
            model_score = -self.grad_V(x_noisy, labels) / self.kT

            # Residual and loss
            residual = model_score - target_score
            loss = np.mean(np.sum(residual ** 2, axis=1))
            total_loss += loss * bs

            # Gradient for shared W
            Wx = x_noisy @ self.W
            grad_W = -2 / self.kT * (residual.T @ Wx) / bs
            grad_W = np.clip(grad_W, -0.1, 0.1)
            self.W -= lr_W * grad_W

            # Gradient for per-class biases
            # ∂loss/∂b_k = -2/kT * mean(residual for class k)
            for k in range(self.n_classes):
                mask = (labels == k)
                if mask.sum() > 0:
                    grad_b_k = -2 * residual[mask].mean(0) / self.kT
                    grad_b_k = np.clip(grad_b_k, -1.0, 1.0)
                    self.b[k] -= lr_b * grad_b_k

        return total_loss / n_samples

    def generate(self, class_idx, n_samples=10, n_steps=500, dt=0.02, anneal=True):
        """Generate samples for a specific class."""
        x = np.random.randn(n_samples, self.n) * 0.5

        if anneal:
            T_start, T_end = 3.0, 0.1
            for step in range(n_steps):
                T = T_start * (T_end / T_start) ** (step / n_steps)
                noise = np.random.randn(*x.shape)
                x = x + self.drift(x, class_idx) * dt + np.sqrt(2 * T * dt) * noise
        else:
            for _ in range(n_steps):
                noise = np.random.randn(*x.shape)
                x = x + self.drift(x, class_idx) * dt + np.sqrt(2 * self.kT * dt) * noise

        return x

    def denoise(self, x_noisy, class_idx, n_steps=100, dt=0.02):
        """Denoise with class-specific potential."""
        x = x_noisy.copy()
        for _ in range(n_steps):
            x = x + self.drift(x, class_idx) * dt + np.sqrt(2 * self.kT * dt * 0.01) * np.random.randn(*x.shape)
        return x

    def classify_by_energy(self, x):
        """Classify by finding which class potential has lowest energy at x."""
        energies = []
        for k in range(self.n_classes):
            # Approximate energy by gradient magnitude (lower = more stable)
            grad = self.grad_V(x, k)
            if x.ndim == 1:
                energy = np.sum(grad ** 2)
            else:
                energy = np.sum(grad ** 2, axis=1)
            energies.append(energy)
        energies = np.array(energies)  # (n_classes, batch) or (n_classes,)
        if energies.ndim == 1:
            return np.argmin(energies)
        return np.argmin(energies, axis=0)


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

    # Scale to [-1, 1]
    X_scaled = 2 * X_train - 1

    n_dim = size * size

    print(f"\nTraining Conditional Criticality Engine ({n_dim} dims, 10 classes)...")
    engine = ConditionalCriticalityEngine(n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=0.5)

    n_epochs = 100
    for epoch in range(n_epochs):
        loss = engine.train_epoch(X_scaled, y_train, lr_b=0.02, lr_W=0.005, sigma=0.3)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.4f}")

    # === Class-conditional Generation ===
    print("\n=== Class-Conditional Generation ===")

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))

    for digit in range(10):
        # Generate 2 samples per digit
        samples = engine.generate(digit, n_samples=2, n_steps=1000, dt=0.02, anneal=True)

        for row in range(2):
            img = ((samples[row] + 1) / 2).reshape(size, size)
            axes[row, digit].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
            axes[row, digit].axis('off')
            if row == 0:
                axes[row, digit].set_title(str(digit))

    plt.suptitle('Conditional Generation: "Generate digit k"', fontsize=12)
    plt.tight_layout()
    plt.savefig('conditional_generated.png', dpi=150)
    print("Saved conditional_generated.png")

    # === Class-conditional Denoising ===
    print("\n=== Class-Conditional Denoising ===")

    test_indices = [np.where(y_train == i)[0][0] for i in range(10)]
    originals = X_scaled[test_indices]
    noisy = originals + np.random.randn(10, n_dim) * 0.3

    # Denoise with correct class label
    denoised_correct = np.zeros_like(noisy)
    for i in range(10):
        denoised_correct[i] = engine.denoise(noisy[i:i+1], class_idx=i, n_steps=200)[0]

    # Denoise with wrong class (use class 0 for all)
    denoised_wrong = engine.denoise(noisy, class_idx=0, n_steps=200)

    fig, axes = plt.subplots(4, 10, figsize=(15, 6))

    for i in range(10):
        orig_img = ((originals[i] + 1) / 2).reshape(size, size)
        noisy_img = ((noisy[i] + 1) / 2).reshape(size, size)
        correct_img = ((denoised_correct[i] + 1) / 2).reshape(size, size)
        wrong_img = ((denoised_wrong[i] + 1) / 2).reshape(size, size)

        axes[0, i].imshow(np.clip(orig_img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(str(i))
        axes[0, i].axis('off')

        axes[1, i].imshow(np.clip(noisy_img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

        axes[2, i].imshow(np.clip(correct_img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')

        axes[3, i].imshow(np.clip(wrong_img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[3, i].axis('off')

    axes[0, 0].set_ylabel('Original', rotation=0, ha='right', fontsize=10)
    axes[1, 0].set_ylabel('Noisy', rotation=0, ha='right', fontsize=10)
    axes[2, 0].set_ylabel('Correct\nclass', rotation=0, ha='right', fontsize=10)
    axes[3, 0].set_ylabel('Wrong\nclass (0)', rotation=0, ha='right', fontsize=10)

    plt.suptitle('Class-Conditional Denoising: Correct vs Wrong Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('conditional_denoise.png', dpi=150)
    print("Saved conditional_denoise.png")

    # === Visualize learned biases ===
    print("\n=== Learned Class Biases ===")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for k in range(10):
        ax = axes[k // 5, k % 5]
        # Bias shows what the class "wants" - negative bias pulls toward +1
        bias_img = ((-engine.b[k] + 1) / 2).reshape(size, size)  # Flip sign to show attractor
        ax.imshow(bias_img, cmap='gray')
        ax.set_title(f'Digit {k}')
        ax.axis('off')

    plt.suptitle('Learned Per-Class Bias Templates (-b_k)', fontsize=12)
    plt.tight_layout()
    plt.savefig('conditional_biases.png', dpi=150)
    print("Saved conditional_biases.png")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
