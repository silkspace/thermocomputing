"""
Coupling Learning via Denoising Score Matching

The PRINCIPLED approach:
- Images are samples from p(x|class) ∝ exp(-x'J_c x / 2)
- J_c is the precision matrix (inverse covariance)
- Learn J_c by matching the score function ∇log p(x) = -J_c x

Denoising score matching:
- Add noise: y = x + ε, where ε ~ N(0, σ²I)
- True score at y (pointing toward data): s(y) ≈ (x - y)/σ² = -ε/σ²
- Model score: s_θ(y) = -J y (for quadratic energy)
- Match: minimize ||s_θ(y) - s(y)||²

This IS trajectory-based learning! The "trajectory" is:
- Start at noisy y
- Observe direction back to clean x
- That direction encodes the energy landscape
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

Path('paper_figures').mkdir(exist_ok=True)


def load_mnist(n_train=5000, n_test=1000, size=14):
    from sklearn.datasets import fetch_openml
    from skimage.transform import resize

    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data.astype(np.float32) / 255.0, mnist.target.astype(int)

    X_small = np.array([resize(x.reshape(28,28), (size,size)).flatten() for x in X])
    return X_small[:n_train], y[:n_train], X_small[-n_test:], y[-n_test:]


def center_train(X, y, n_classes=10):
    """Center training data, return means for test."""
    X_c = X.copy()
    means = {}
    for c in range(n_classes):
        mask = y == c
        if mask.sum():
            means[c] = X[mask].mean(0)
            X_c[mask] -= means[c]
    return X_c, means


class ScoreMatchingLearner:
    """Learn J via denoising score matching."""

    def __init__(self, n_dim: int, n_classes: int, sigma: float = 0.1):
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.sigma = sigma
        self.J = np.zeros((n_classes, n_dim, n_dim))

    def score(self, y: np.ndarray, c: int) -> np.ndarray:
        """Model score: s(y) = -J_c y"""
        return -self.J[c] @ y

    def denoise_direction(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """True score (denoising direction): (x - y) / σ²"""
        return (x - y) / (self.sigma ** 2)

    def compute_gradient(self, x: np.ndarray, c: int) -> np.ndarray:
        """
        Score matching gradient for one sample.

        Add noise: y = x + ε
        True score: s_true = -ε/σ²
        Model score: s_model = -J y

        Loss: ||s_model - s_true||²
        Gradient: ∂L/∂J = 2(s_model - s_true) ⊗ (-y)
                        = -2(J y + ε/σ²) y'
        """
        # Add noise
        eps = np.random.randn(self.n_dim) * self.sigma
        y = x + eps

        # Scores
        s_true = -eps / (self.sigma ** 2)
        s_model = -self.J[c] @ y

        # Residual
        residual = s_model - s_true  # = -J y + ε/σ²

        # Gradient: minimize ||residual||² w.r.t. J
        # ∂||s_model - s_true||²/∂J = 2 * residual * ∂s_model/∂J
        # s_model = -J y, so ∂s_model/∂J[i,k] = -y[k] (for row i)
        # Full gradient: -2 * residual ⊗ y' (outer product)
        grad = -2 * np.outer(residual, y)

        return grad

    def energy(self, x: np.ndarray, c: int) -> float:
        return 0.5 * x @ self.J[c] @ x

    def classify_with_means(self, x_raw: np.ndarray, means: Dict) -> int:
        """Classify by minimum energy, trying all class hypotheses."""
        energies = []
        for c in range(self.n_classes):
            x_c = x_raw - means.get(c, 0)
            energies.append(self.energy(x_c, c))
        return int(np.argmin(energies))


def train_score_matching(
    learner: ScoreMatchingLearner,
    X_train: np.ndarray,
    y_train: np.ndarray,
    means: Dict,
    n_epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 500,
    n_noise_samples: int = 5,
    X_test_raw: np.ndarray = None,
    y_test: np.ndarray = None
):
    """Train via denoising score matching."""
    n_samples = len(X_train)
    n_classes = learner.n_classes

    print(f"\nScore Matching Training: {n_epochs} epochs, σ={learner.sigma}")

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)[:batch_size]

        grad_accum = np.zeros_like(learner.J)
        counts = np.zeros(n_classes)

        for idx in perm:
            x = X_train[idx]
            c = int(y_train[idx])

            # Average gradient over multiple noise samples
            sample_grad = np.zeros((learner.n_dim, learner.n_dim))
            for _ in range(n_noise_samples):
                sample_grad += learner.compute_gradient(x, c)
            sample_grad /= n_noise_samples

            grad_accum[c] += sample_grad
            counts[c] += 1

        # Update
        for c in range(n_classes):
            if counts[c] > 0:
                avg_grad = grad_accum[c] / counts[c]
                # Gradient descent on loss
                learner.J[c] -= lr * avg_grad
                # Enforce symmetry
                learner.J[c] = (learner.J[c] + learner.J[c].T) / 2

        if epoch % 10 == 0:
            # Test accuracy (no label leak)
            if X_test_raw is not None:
                correct = sum(1 for i in range(len(X_test_raw))
                              if learner.classify_with_means(X_test_raw[i], means) == y_test[i])
                test_acc = correct / len(X_test_raw) * 100
            else:
                test_acc = 0
            print(f"Epoch {epoch:3d}: Test={test_acc:5.1f}%, ||J||={np.linalg.norm(learner.J):.4f}")

    return learner


def main():
    print("=" * 70)
    print("COUPLING LEARNING: Denoising Score Matching")
    print("=" * 70)

    size = 14
    n_dim = size ** 2

    X_train_raw, y_train, X_test_raw, y_test = load_mnist(5000, 1000, size)

    # Baseline
    centroids = np.array([X_train_raw[y_train == c].mean(0) for c in range(10)])
    nc_raw = np.mean([np.argmin(np.linalg.norm(centroids - x, axis=1)) == y
                       for x, y in zip(X_test_raw, y_test)]) * 100
    print(f"Nearest Centroid (raw): {nc_raw:.1f}%")

    # Center
    X_train, means = center_train(X_train_raw, y_train)

    # Verify centering kills NC
    X_test_oracle = X_test_raw.copy()
    for c in range(10):
        mask = y_test == c
        X_test_oracle[mask] -= means[c]
    centroids_c = np.array([X_train[y_train == c].mean(0) for c in range(10)])
    nc_centered = np.mean([np.argmin(np.linalg.norm(centroids_c - x, axis=1)) == y
                            for x, y in zip(X_test_oracle, y_test)]) * 100
    print(f"Nearest Centroid (centered, oracle): {nc_centered:.1f}%")

    # Train
    learner = ScoreMatchingLearner(n_dim, 10, sigma=0.3)
    train_score_matching(
        learner, X_train, y_train, means,
        n_epochs=100,
        lr=0.01,
        batch_size=500,
        n_noise_samples=10,
        X_test_raw=X_test_raw,
        y_test=y_test
    )

    # Final
    final_acc = sum(1 for i in range(len(X_test_raw))
                    if learner.classify_with_means(X_test_raw[i], means) == y_test[i]) / len(X_test_raw) * 100
    print(f"\nFinal: {final_acc:.1f}%")

    # Compare to QDA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
    qda.fit(X_train_raw, y_train)
    qda_acc = qda.score(X_test_raw, y_test) * 100
    print(f"QDA: {qda_acc:.1f}%")


if __name__ == "__main__":
    main()
