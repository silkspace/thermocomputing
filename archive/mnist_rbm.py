"""
Restricted Boltzmann Machine for MNIST.

This adds HIDDEN units - the true "latent space" that captures
digit identity and style variations.

Architecture:
- Visible: 64 pixels (8x8 downsampled MNIST)
- Hidden: 32 latent units (learns digit features)
- No V-V or H-H connections (restricted)
- Only V-H connections (bipartite)
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


def downsample(images, size=8):
    n = len(images)
    images = images.reshape(n, 28, 28)
    factor = 28 // size
    downsampled = np.zeros((n, size, size))
    for i in range(size):
        for j in range(size):
            downsampled[:, i, j] = images[:,
                i*factor:(i+1)*factor,
                j*factor:(j+1)*factor
            ].mean(axis=(1, 2))
    return downsampled.reshape(n, size * size)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class RBM:
    """
    Restricted Boltzmann Machine.

    Energy: E(v,h) = -b'v - c'h - v'Wh

    where:
        v = visible units (pixels)
        h = hidden units (latent features)
        b = visible biases
        c = hidden biases
        W = visible-hidden weights
    """

    def __init__(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Initialize parameters
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.b = np.zeros(n_visible)  # visible biases
        self.c = np.zeros(n_hidden)   # hidden biases

    def sample_hidden(self, v):
        """Sample h given v. P(h_j=1|v) = sigmoid(c_j + sum_i W_ij v_i)"""
        activation = self.c + v @ self.W
        prob = sigmoid(activation)
        return prob, (np.random.rand(*prob.shape) < prob).astype(np.float32)

    def sample_visible(self, h):
        """Sample v given h. P(v_i=1|h) = sigmoid(b_i + sum_j W_ij h_j)"""
        activation = self.b + h @ self.W.T
        prob = sigmoid(activation)
        return prob, (np.random.rand(*prob.shape) < prob).astype(np.float32)

    def gibbs_step(self, v):
        """One step of Gibbs sampling: v -> h -> v'"""
        h_prob, h_sample = self.sample_hidden(v)
        v_prob, v_sample = self.sample_visible(h_sample)
        return h_prob, h_sample, v_prob, v_sample

    def contrastive_divergence(self, v_data, k=1, lr=0.01):
        """
        CD-k learning step.

        Gradient of log-likelihood:
            ∂log P(v)/∂W_ij = <v_i h_j>_data - <v_i h_j>_model
            ∂log P(v)/∂b_i = <v_i>_data - <v_i>_model
            ∂log P(v)/∂c_j = <h_j>_data - <h_j>_model
        """
        batch_size = len(v_data)

        # Positive phase (data)
        h_prob_data, h_sample = self.sample_hidden(v_data)

        # Negative phase (k steps of Gibbs)
        v_sample = v_data
        for _ in range(k):
            h_prob, h_sample, v_prob, v_sample = self.gibbs_step(v_sample)
        h_prob_model = h_prob
        v_model = v_sample

        # Compute gradients
        # <v_i h_j>_data ≈ v_data' @ h_prob_data / batch_size
        # <v_i h_j>_model ≈ v_model' @ h_prob_model / batch_size
        pos_associations = v_data.T @ h_prob_data / batch_size
        neg_associations = v_model.T @ h_prob_model / batch_size

        # Update weights and biases
        self.W += lr * (pos_associations - neg_associations)
        self.b += lr * (v_data.mean(axis=0) - v_model.mean(axis=0))
        self.c += lr * (h_prob_data.mean(axis=0) - h_prob_model.mean(axis=0))

        # Return reconstruction error
        recon_error = np.mean((v_data - v_prob) ** 2)
        return recon_error

    def reconstruct(self, v):
        """Reconstruct v through hidden layer."""
        h_prob, _ = self.sample_hidden(v)
        v_prob, _ = self.sample_visible(h_prob)
        return v_prob

    def generate(self, n_samples, n_gibbs=1000):
        """Generate samples by running Gibbs sampling."""
        v = (np.random.rand(n_samples, self.n_visible) > 0.5).astype(np.float32)
        for _ in range(n_gibbs):
            _, _, v_prob, v = self.gibbs_step(v)
        return v_prob, v

    def get_hidden_representation(self, v):
        """Get hidden unit activations for visualization."""
        h_prob, _ = self.sample_hidden(v)
        return h_prob


def main():
    # Load data
    X, y = load_mnist()

    # Use all digits, 500 samples each
    n_per_digit = 500
    X_train = []
    y_train = []
    for digit in range(10):
        mask = (y == digit)
        X_digit = X[mask][:n_per_digit]
        X_train.append(X_digit)
        y_train.extend([digit] * len(X_digit))
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    print(f"Training data: {len(X_train)} samples")

    # Preprocess
    size = 8
    X_train = downsample(X_train, size=size)
    X_train = (X_train > 0.3).astype(np.float32)  # Binarize

    n_visible = size * size
    n_hidden = 64  # Latent dimensions

    print(f"Visible units: {n_visible}")
    print(f"Hidden units: {n_hidden}")

    # Show training samples
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for digit in range(10):
        idx = np.where(y_train == digit)[0][0]
        axes[0, digit].imshow(X_train[idx].reshape(size, size), cmap='gray')
        axes[0, digit].set_title(str(digit))
        axes[0, digit].axis('off')

        idx = np.where(y_train == digit)[0][1]
        axes[1, digit].imshow(X_train[idx].reshape(size, size), cmap='gray')
        axes[1, digit].axis('off')
    plt.suptitle('Training Data (all digits)')
    plt.savefig('rbm_data.png', dpi=150)
    print("Saved rbm_data.png")

    # Create and train RBM
    rbm = RBM(n_visible, n_hidden)

    n_epochs = 50
    batch_size = 100
    k_cd = 5
    lr = 0.1

    errors = []
    print("\nTraining RBM...")

    for epoch in range(n_epochs):
        perm = np.random.permutation(len(X_train))
        X_shuffled = X_train[perm]

        epoch_error = 0
        n_batches = len(X_train) // batch_size

        for b in range(n_batches):
            batch = X_shuffled[b*batch_size:(b+1)*batch_size]
            error = rbm.contrastive_divergence(batch, k=k_cd, lr=lr)
            epoch_error += error

        epoch_error /= n_batches
        errors.append(epoch_error)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}: recon_error = {epoch_error:.4f}")

    print("\nTraining complete!")

    # Generate samples
    print("\nGenerating samples...")
    _, generated = rbm.generate(n_samples=20, n_gibbs=2000)

    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(20):
        ax = axes[i // 10, i % 10]
        ax.imshow(generated[i].reshape(size, size), cmap='gray')
        ax.axis('off')
    plt.suptitle('Generated Samples (from learned RBM)')
    plt.savefig('rbm_generated.png', dpi=150)
    print("Saved rbm_generated.png")

    # Reconstructions
    print("\nComputing reconstructions...")
    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5))
    for digit in range(10):
        idx = np.where(y_train == digit)[0][0]
        original = X_train[idx]
        recon = rbm.reconstruct(original.reshape(1, -1))[0]

        axes[0, digit].imshow(original.reshape(size, size), cmap='gray')
        axes[0, digit].set_title(str(digit))
        axes[0, digit].axis('off')

        axes[1, digit].imshow(recon.reshape(size, size), cmap='gray')
        axes[1, digit].axis('off')

        axes[2, digit].imshow(np.abs(original - recon).reshape(size, size), cmap='Reds')
        axes[2, digit].axis('off')

    axes[0, 0].set_ylabel('Original', rotation=0, ha='right')
    axes[1, 0].set_ylabel('Recon', rotation=0, ha='right')
    axes[2, 0].set_ylabel('Error', rotation=0, ha='right')
    plt.suptitle('Reconstructions through Hidden Layer')
    plt.savefig('rbm_recon.png', dpi=150)
    print("Saved rbm_recon.png")

    # Visualize hidden units (receptive fields)
    print("\nVisualizing learned features...")
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(n_hidden):
        ax = axes[i // 8, i % 8]
        weights = rbm.W[:, i].reshape(size, size)
        ax.imshow(weights, cmap='RdBu', vmin=-np.abs(weights).max(), vmax=np.abs(weights).max())
        ax.axis('off')
    plt.suptitle('Learned Features (Hidden Unit Receptive Fields)')
    plt.savefig('rbm_features.png', dpi=150)
    print("Saved rbm_features.png")

    # Training curve
    plt.figure(figsize=(8, 4))
    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.title('RBM Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('rbm_training.png', dpi=150)
    print("Saved rbm_training.png")

    # Hidden space visualization
    print("\nComputing hidden representations...")
    h_reps = []
    labels = []
    for digit in range(10):
        mask = (y_train == digit)
        X_digit = X_train[mask][:50]
        h = rbm.get_hidden_representation(X_digit)
        h_reps.append(h)
        labels.extend([digit] * len(h))
    h_reps = np.vstack(h_reps)
    labels = np.array(labels)

    # Simple 2D projection using PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    h_2d = pca.fit_transform(h_reps)

    plt.figure(figsize=(10, 8))
    for digit in range(10):
        mask = (labels == digit)
        plt.scatter(h_2d[mask, 0], h_2d[mask, 1], label=str(digit), alpha=0.6)
    plt.legend()
    plt.title('Hidden Space (PCA projection)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('rbm_latent.png', dpi=150)
    print("Saved rbm_latent.png")

    print("\n=== Summary ===")
    print(f"Learned {n_visible}x{n_hidden} = {n_visible * n_hidden} weights")
    print(f"Plus {n_visible} visible biases and {n_hidden} hidden biases")
    print(f"Total parameters: {n_visible * n_hidden + n_visible + n_hidden}")
    print(f"Final reconstruction error: {errors[-1]:.4f}")
    print("\nThe hidden units are the LATENT SPACE.")
    print("Each hidden unit learns a feature (stroke, curve, position).")
    print("Digits are represented as combinations of these features.")


if __name__ == "__main__":
    main()
