"""
Denoising experiment: show thermodynamic relaxation cleaning up noisy digits.

This is the key demo: start with corrupted input, let the system relax,
watch it converge to a clean digit.
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class RBM:
    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.b = np.zeros(n_visible)
        self.c = np.zeros(n_hidden)

    def hidden_prob(self, v):
        return sigmoid(self.c + v @ self.W)

    def visible_prob(self, h):
        return sigmoid(self.b + h @ self.W.T)

    def gibbs_step(self, v, sample=True):
        """v -> h -> v', return probabilities and samples"""
        h_prob = self.hidden_prob(v)
        h = (np.random.rand(*h_prob.shape) < h_prob) if sample else h_prob
        v_prob = self.visible_prob(h)
        v_new = (np.random.rand(*v_prob.shape) < v_prob) if sample else v_prob
        return h_prob, h, v_prob, v_new

    def denoise(self, v_noisy, n_steps=50, return_trajectory=False):
        """Denoise by running Gibbs sampling, return probability (grayscale)"""
        v = v_noisy.copy()
        trajectory = [v.copy()] if return_trajectory else None

        for _ in range(n_steps):
            h_prob = self.hidden_prob(v)
            # Use probabilities (mean-field) for smoother output
            v_prob = self.visible_prob(h_prob)
            # Mix: gradually trust the model more
            v = v_prob
            if return_trajectory:
                trajectory.append(v.copy())

        return (v, trajectory) if return_trajectory else v

    def train(self, X, n_epochs=30, batch_size=100, lr=0.1, k=1):
        n_samples = len(X)
        for epoch in range(n_epochs):
            perm = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch = X[perm[i:i+batch_size]]
                self._cd_step(batch, lr, k)
            if epoch % 10 == 0:
                recon = self.visible_prob(self.hidden_prob(X[:100]))
                error = np.mean((X[:100] - recon) ** 2)
                print(f"  Epoch {epoch}: recon_error = {error:.4f}")

    def _cd_step(self, v_data, lr, k):
        batch_size = len(v_data)
        h_prob_data = self.hidden_prob(v_data)

        v = v_data.copy()
        for _ in range(k):
            _, _, _, v = self.gibbs_step(v)
        h_prob_model = self.hidden_prob(v)

        self.W += lr * (v_data.T @ h_prob_data - v.T @ h_prob_model) / batch_size
        self.b += lr * (v_data.mean(0) - v.mean(0))
        self.c += lr * (h_prob_data.mean(0) - h_prob_model.mean(0))


def add_noise(images, noise_level=0.3):
    """Add salt-and-pepper noise"""
    noisy = images.copy()
    mask = np.random.rand(*images.shape) < noise_level
    noisy[mask] = 1 - noisy[mask]  # Flip pixels
    return noisy


def main():
    X, y = load_mnist()

    # Use 14x14 for better quality (196 pixels)
    size = 14
    print(f"Using {size}x{size} = {size*size} pixels")

    # Downsample
    n_train = 5000
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

    # Keep as grayscale (not binary)
    print(f"Training data range: [{X_train.min():.2f}, {X_train.max():.2f}]")

    n_visible = size * size
    n_hidden = 500  # More capacity for sharper representations

    # Train RBM
    print(f"\nTraining RBM ({n_visible} visible, {n_hidden} hidden)...")
    rbm = RBM(n_visible, n_hidden)
    rbm.train(X_train, n_epochs=100, batch_size=100, lr=0.05, k=10)

    # === DENOISING EXPERIMENT ===
    print("\n=== Denoising Experiment ===")

    # Pick one example of each digit
    test_indices = []
    for digit in range(10):
        idx = np.where(y_train == digit)[0][0]
        test_indices.append(idx)

    originals = X_train[test_indices]
    noisy = add_noise(originals, noise_level=0.3)
    denoised = rbm.denoise(noisy, n_steps=100)

    # Plot: Original | Noisy | Denoised
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))

    for i in range(10):
        axes[0, i].imshow(originals[i].reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(str(i))

        axes[1, i].imshow(noisy[i].reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

        axes[2, i].imshow(denoised[i].reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')

    axes[0, 0].set_ylabel('Original', rotation=0, ha='right', fontsize=12)
    axes[1, 0].set_ylabel('Noisy', rotation=0, ha='right', fontsize=12)
    axes[2, 0].set_ylabel('Denoised', rotation=0, ha='right', fontsize=12)

    plt.suptitle('Thermodynamic Denoising: Relaxation Cleans Corrupted Digits', fontsize=14)
    plt.tight_layout()
    plt.savefig('denoise_result.png', dpi=150)
    print("Saved denoise_result.png")

    # === RELAXATION TRAJECTORY ===
    print("\n=== Relaxation Trajectory ===")

    # Pick a single noisy digit and show its relaxation over time
    digit_idx = 3  # Use digit '3'
    idx = np.where(y_train == digit_idx)[0][0]
    original = X_train[idx:idx+1]
    noisy_single = add_noise(original, noise_level=0.4)

    _, trajectory = rbm.denoise(noisy_single, n_steps=100, return_trajectory=True)

    # Show trajectory at selected time points
    steps_to_show = [0, 1, 2, 5, 10, 20, 50, 100]
    fig, axes = plt.subplots(1, len(steps_to_show) + 1, figsize=(15, 2))

    # Original
    axes[0].imshow(original[0].reshape(size, size), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i, step in enumerate(steps_to_show):
        axes[i+1].imshow(trajectory[step][0].reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[i+1].set_title(f't={step}')
        axes[i+1].axis('off')

    plt.suptitle(f'Relaxation Trajectory: Noisy → Clean (digit {digit_idx})', fontsize=12)
    plt.tight_layout()
    plt.savefig('denoise_trajectory.png', dpi=150)
    print("Saved denoise_trajectory.png")

    # === GENERATION (grayscale, better quality) ===
    print("\n=== Generation ===")

    # Generate by running long Gibbs chains
    n_gen = 20
    v = np.random.rand(n_gen, n_visible)  # Random init
    for _ in range(2000):
        h_prob = rbm.hidden_prob(v)
        h = (np.random.rand(*h_prob.shape) < h_prob)
        v_prob = rbm.visible_prob(h)
        v = (np.random.rand(*v_prob.shape) < v_prob)

    # Get final probabilities (grayscale)
    h_prob = rbm.hidden_prob(v)
    v_final = rbm.visible_prob(h_prob)

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(20):
        ax = axes[i // 10, i % 10]
        ax.imshow(v_final[i].reshape(size, size), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    plt.suptitle('Generated Digits (Grayscale, from Thermal Equilibrium)', fontsize=12)
    plt.tight_layout()
    plt.savefig('generated_grayscale.png', dpi=150)
    print("Saved generated_grayscale.png")

    # === RECONSTRUCTION QUALITY ===
    print("\n=== Reconstruction Quality ===")

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        original = X_train[idx]
        recon = rbm.visible_prob(rbm.hidden_prob(original.reshape(1, -1)))[0]

        axes[0, i].imshow(original.reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(str(i))
        axes[0, i].axis('off')

        axes[1, i].imshow(recon.reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Input', rotation=0, ha='right')
    axes[1, 0].set_ylabel('Recon', rotation=0, ha='right')
    plt.suptitle('Reconstruction through Latent Space', fontsize=12)
    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=150)
    print("Saved reconstruction.png")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
