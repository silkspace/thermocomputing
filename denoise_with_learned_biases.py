"""
Denoising demo using learned biases from trajectory estimation.

This script shows that the biases we learned via trajectory estimation
can be used to denoise digits - i.e., they encode the attractor basins
for each digit class.

Process:
1. Load learned biases
2. Start from noisy/corrupted digit
3. Run Langevin dynamics under φ⁴ + bias potential
4. Watch digit converge to clean version
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class PhiQuarticWithBias:
    """φ⁴ + bias potential with learned class-specific biases."""

    def __init__(self, biases: np.ndarray, J2: float = -2.0, J4: float = 1.0, kT: float = 0.3, bias_scale: float = 1.0):
        """
        Args:
            biases: Shape (n_classes, n_dim), learned biases for each class
            bias_scale: Scale factor for biases (increase for stronger attraction)
        """
        self.biases = biases * bias_scale
        self.J2 = J2
        self.J4 = J4
        self.kT = kT

    def grad_V(self, x: np.ndarray, class_idx: int) -> np.ndarray:
        """∂V/∂x = 2J₂x + 4J₄x³ + b_c"""
        return 2 * self.J2 * x + 4 * self.J4 * (x ** 3) + self.biases[class_idx]

    def langevin_step(self, x: np.ndarray, class_idx: int, dt: float = 0.05) -> np.ndarray:
        """One Langevin step: dx = -∇V·dt + √(2kT·dt)·η"""
        grad = self.grad_V(x, class_idx)
        noise = np.sqrt(2 * self.kT * dt) * np.random.randn(*x.shape)
        return x - grad * dt + noise


def load_digit_samples(digit: int, n_samples: int = 5, image_size: int = 14):
    """Load sample digits from MNIST."""
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    mask = (y == digit)
    digit_images = X[mask][:n_samples]

    # Downsample
    X_reshaped = digit_images.reshape(-1, 28, 28)
    factor = 28 // image_size
    X_down = np.zeros((len(digit_images), image_size, image_size))

    for i in range(image_size):
        for j in range(image_size):
            X_down[:, i, j] = X_reshaped[:,
                i*factor:(i+1)*factor,
                j*factor:(j+1)*factor
            ].mean(axis=(1, 2))

    # Scale to [-1, 1]
    X_scaled = 2 * X_down.reshape(n_samples, -1) - 1

    return X_scaled


def denoise_trajectory(model, x_init, class_idx, n_steps=100, dt=0.05):
    """Run Langevin dynamics to denoise."""
    trajectory = [x_init.copy()]
    x = x_init.copy()

    for _ in range(n_steps):
        x = model.langevin_step(x, class_idx, dt)
        trajectory.append(x.copy())

    return trajectory


def main():
    image_size = 14
    n_dim = image_size * image_size

    # Load learned biases
    bias_path = Path('learned_biases.npy')
    if not bias_path.exists():
        print("No learned_biases.npy found. Running trajectory_estimator first...")
        from trajectory_estimator import main as train_main
        train_main()

    biases = np.load('learned_biases.npy')
    print(f"Loaded biases shape: {biases.shape}")

    # Create model for denoising (moderate bias)
    model = PhiQuarticWithBias(biases, J2=-2.0, J4=1.0, kT=0.3, bias_scale=3.0)

    # Demo: denoise different digits
    digits_to_show = [3, 7, 0, 5]
    steps_to_show = [0, 2, 5, 10, 20, 40, 60, 100]

    fig, axes = plt.subplots(len(digits_to_show), len(steps_to_show), figsize=(16, 8))

    for row, digit in enumerate(digits_to_show):
        print(f"Denoising digit {digit}...")

        # Load a real digit sample
        samples = load_digit_samples(digit, n_samples=1, image_size=image_size)
        x_clean = samples[0]

        # Corrupt with noise
        np.random.seed(42 + digit)
        x_noisy = x_clean + np.random.randn(n_dim) * 0.8  # Heavy noise

        # Run denoising
        trajectory = denoise_trajectory(model, x_noisy, class_idx=digit, n_steps=100, dt=0.05)

        # Plot trajectory
        for col, step in enumerate(steps_to_show):
            ax = axes[row, col]
            img = ((trajectory[step] + 1) / 2).reshape(image_size, image_size)
            ax.imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
            if row == 0:
                ax.set_title(f't={step}', fontsize=11)
            if col == 0:
                ax.set_ylabel(f'Digit {digit}', fontsize=11)
            ax.axis('off')

    plt.suptitle('Denoising with Learned φ⁴+bias Potential (Trajectory Estimation)', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('paper_figures/denoising_learned_biases.png', dpi=150, bbox_inches='tight')
    print("Saved paper_figures/denoising_learned_biases.png")

    # Also create single-digit comparison figure (like Figure 5)
    print("\nCreating Figure 5 replacement...")
    digit = 3

    # For generation from noise, need stronger bias
    model_strong = PhiQuarticWithBias(biases, J2=-2.0, J4=1.0, kT=0.2, bias_scale=5.0)

    np.random.seed(42)
    x_noisy = np.random.randn(n_dim) * 0.5  # Start from pure noise

    trajectory = denoise_trajectory(model_strong, x_noisy, class_idx=digit, n_steps=100, dt=0.05)

    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(16, 2.2))

    for i, step in enumerate(steps_to_show):
        img = ((trajectory[step] + 1) / 2).reshape(image_size, image_size)
        axes[i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f't={step}', fontsize=11)
        axes[i].axis('off')

    plt.suptitle('Relaxation under φ⁴+bias Potential: Fast Convergence to Attractor', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('paper_figures/trajectory_fast_learned.png', dpi=150, bbox_inches='tight')
    print("Saved paper_figures/trajectory_fast_learned.png")


if __name__ == "__main__":
    main()
