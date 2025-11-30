"""
Learn the latent structure of MNIST using thermodynamic learning rules.

We'll start simple:
- Downsample MNIST to 8x8 = 64 pixels
- Binarize pixels
- Learn an Ising model (biases + pairwise couplings)
- Use the finite-time trajectory likelihood learning rule
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Download MNIST if needed
def load_mnist():
    """Load MNIST using sklearn (simpler than torchvision for this)."""
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int32)
        return X, y
    except Exception as e:
        print(f"Error loading MNIST: {e}")
        print("Generating synthetic data instead...")
        return None, None


def downsample(images, size=8):
    """Downsample 28x28 images to size x size."""
    n = len(images)
    images = images.reshape(n, 28, 28)

    # Simple averaging downsample
    factor = 28 // size
    downsampled = np.zeros((n, size, size))
    for i in range(size):
        for j in range(size):
            downsampled[:, i, j] = images[:,
                i*factor:(i+1)*factor,
                j*factor:(j+1)*factor
            ].mean(axis=(1, 2))

    return downsampled.reshape(n, size * size)


def binarize(images, threshold=0.5):
    """Convert to binary (±1 for Ising model)."""
    return 2.0 * (images > threshold).astype(np.float32) - 1.0


class IsingModel:
    """Simple Ising model with biases and pairwise couplings."""

    def __init__(self, n_visible, n_hidden=0):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_total = n_visible + n_hidden

        # Parameters
        self.b = np.zeros(self.n_total)  # biases
        # Sparse couplings (only nearest neighbors in 2D grid + some random)
        self.J = np.zeros((self.n_total, self.n_total))

    def energy(self, s):
        """Compute energy E(s) = -sum_i b_i s_i - sum_{i<j} J_ij s_i s_j"""
        return -np.dot(self.b, s) - 0.5 * s @ self.J @ s

    def local_field(self, s, i):
        """Compute local field h_i = b_i + sum_j J_ij s_j"""
        return self.b[i] + np.dot(self.J[i], s)

    def gibbs_step(self, s, beta=1.0):
        """One Gibbs sampling sweep (update all spins once in random order)."""
        s = s.copy()
        order = np.random.permutation(self.n_total)
        for i in order:
            h_i = self.local_field(s, i)
            p_plus = 1.0 / (1.0 + np.exp(-2 * beta * h_i))
            s[i] = 1.0 if np.random.rand() < p_plus else -1.0
        return s

    def sample(self, n_samples, n_steps=100, beta=1.0):
        """Generate samples via Gibbs sampling."""
        samples = []
        s = np.random.choice([-1.0, 1.0], size=self.n_total)
        for _ in range(n_steps):  # burn-in
            s = self.gibbs_step(s, beta)
        for _ in range(n_samples):
            for _ in range(10):  # steps between samples
                s = self.gibbs_step(s, beta)
            samples.append(s.copy())
        return np.array(samples)


def contrastive_divergence_step(model, data_batch, k=1, lr_b=0.01, lr_J=0.001, beta=1.0):
    """
    One step of Contrastive Divergence learning.

    This is the classic Boltzmann machine learning rule, which is related to
    the trajectory-based rule: we compare data statistics to model statistics.
    """
    batch_size = len(data_batch)
    n = model.n_total

    # Positive phase: clamp to data
    # For visible-only model, data is the full state
    v_data = data_batch

    # Compute data statistics
    data_mean = np.mean(v_data, axis=0)  # <s_i>_data
    data_corr = np.zeros((n, n))
    for v in v_data:
        data_corr += np.outer(v, v)
    data_corr /= batch_size  # <s_i s_j>_data

    # Negative phase: run k steps of Gibbs from data
    v_model = v_data.copy()
    for _ in range(k):
        for i in range(batch_size):
            v_model[i] = model.gibbs_step(v_model[i], beta)

    # Compute model statistics
    model_mean = np.mean(v_model, axis=0)  # <s_i>_model
    model_corr = np.zeros((n, n))
    for v in v_model:
        model_corr += np.outer(v, v)
    model_corr /= batch_size  # <s_i s_j>_model

    # Update parameters: gradient ascent on log-likelihood
    # ΔΔb_i ∝ <s_i>_data - <s_i>_model
    # ΔJ_ij ∝ <s_i s_j>_data - <s_i s_j>_model
    model.b += lr_b * (data_mean - model_mean)

    dJ = lr_J * (data_corr - model_corr)
    dJ = 0.5 * (dJ + dJ.T)  # Symmetrize
    np.fill_diagonal(dJ, 0)  # No self-coupling
    model.J += dJ

    # Return reconstruction error for monitoring
    recon_error = np.mean((v_data - v_model) ** 2)
    return recon_error


def main():
    # Load and preprocess MNIST
    X, y = load_mnist()

    if X is None:
        print("Using synthetic digit-like patterns...")
        # Create simple synthetic patterns
        n_samples = 1000
        size = 8
        X = np.zeros((n_samples, size * size))
        for i in range(n_samples):
            img = np.zeros((size, size))
            # Random vertical or horizontal bar
            if np.random.rand() > 0.5:
                col = np.random.randint(1, size-1)
                img[:, col] = 1
            else:
                row = np.random.randint(1, size-1)
                img[row, :] = 1
            X[i] = img.flatten()
        X = binarize(X, threshold=0.5)
    else:
        # Use just digit '1' for simplicity (clearest structure)
        digit = 1
        mask = (y == digit)
        X = X[mask][:1000]  # Take 1000 samples

        print(f"Using {len(X)} samples of digit {digit}")

        # Downsample and binarize
        size = 8
        X = downsample(X, size=size)
        X = binarize(X, threshold=0.3)

    n_visible = X.shape[1]
    print(f"Data shape: {X.shape}")
    print(f"Visible units: {n_visible}")

    # Show some data samples
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        img = X[i].reshape(int(np.sqrt(n_visible)), -1)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.suptitle('Training Data (downsampled, binarized)')
    plt.savefig('mnist_data.png', dpi=150)
    print("Saved mnist_data.png")

    # Create and train Ising model
    model = IsingModel(n_visible)

    # Training
    n_epochs = 100
    batch_size = 50
    k_cd = 5  # CD-k steps

    errors = []
    print("\nTraining Ising model...")

    for epoch in range(n_epochs):
        # Shuffle data
        perm = np.random.permutation(len(X))
        X_shuffled = X[perm]

        epoch_error = 0
        n_batches = len(X) // batch_size

        for b in range(n_batches):
            batch = X_shuffled[b*batch_size:(b+1)*batch_size]
            error = contrastive_divergence_step(
                model, batch, k=k_cd,
                lr_b=0.1, lr_J=0.01, beta=1.0
            )
            epoch_error += error

        epoch_error /= n_batches
        errors.append(epoch_error)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: recon_error = {epoch_error:.4f}")

    print("\nTraining complete!")

    # Generate samples from learned model
    print("\nGenerating samples from learned distribution...")
    samples = model.sample(n_samples=10, n_steps=500, beta=1.0)

    # Show generated samples
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        img = samples[i, :n_visible].reshape(int(np.sqrt(n_visible)), -1)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
    plt.suptitle('Generated Samples (from learned Ising model)')
    plt.savefig('mnist_generated.png', dpi=150)
    print("Saved mnist_generated.png")

    # Plot training curve
    plt.figure(figsize=(8, 4))
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('mnist_training.png', dpi=150)
    print("Saved mnist_training.png")

    # Visualize learned parameters
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Biases as image
    bias_img = model.b.reshape(int(np.sqrt(n_visible)), -1)
    im = axes[0].imshow(bias_img, cmap='RdBu', vmin=-np.abs(bias_img).max(), vmax=np.abs(bias_img).max())
    axes[0].set_title('Learned Biases')
    plt.colorbar(im, ax=axes[0])

    # Coupling matrix
    im = axes[1].imshow(model.J, cmap='RdBu', vmin=-np.abs(model.J).max(), vmax=np.abs(model.J).max())
    axes[1].set_title('Learned Couplings')
    plt.colorbar(im, ax=axes[1])

    plt.tight_layout()
    plt.savefig('mnist_params.png', dpi=150)
    print("Saved mnist_params.png")

    print("\n=== Summary ===")
    print(f"Learned {n_visible} biases and {n_visible*(n_visible-1)//2} couplings")
    print(f"Final reconstruction error: {errors[-1]:.4f}")
    print("\nThe learned Ising model captures the statistical structure of the digit.")
    print("This is the 'latent distribution' - the energy landscape whose")
    print("equilibrium distribution matches the data distribution.")


if __name__ == "__main__":
    main()
