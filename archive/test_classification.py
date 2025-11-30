"""
Test energy-based classification: k* = argmin_k V_k(x)

The same potentials that enable generation should enable recognition.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mnist_conditional import ConditionalCriticalityEngine, load_mnist


def compute_energy(engine, x, class_idx):
    """Compute V_k(x) = J2*||x||² + J4*||x||⁴ + b_k·x + 0.5*x^T J x"""
    # Per-element terms
    energy = engine.J2 * np.sum(x**2) + engine.J4 * np.sum(x**4)
    # Bias term
    energy += np.dot(engine.b[class_idx], x)
    # Coupling term: 0.5 * x^T W W^T x = 0.5 * ||W^T x||²
    Wx = engine.W.T @ x
    energy += 0.5 * np.dot(Wx, Wx)
    return energy


def compute_grad_norm(engine, x, class_idx):
    """Compute ||∇V_k(x)||²"""
    grad = engine.grad_V(x, class_idx)
    return np.sum(grad**2)


def classify_by_energy(engine, x):
    """Classify x by finding which class potential has lowest energy."""
    energies = [compute_energy(engine, x, k) for k in range(engine.n_classes)]
    return np.argmin(energies), energies


def classify_by_gradient(engine, x):
    """Classify x by finding which class has smallest gradient (closest to equilibrium)."""
    grad_norms = [compute_grad_norm(engine, x, k) for k in range(engine.n_classes)]
    return np.argmin(grad_norms), grad_norms


def main():
    X, y = load_mnist()

    size = 14
    n_train = 5000
    n_test = 1000

    # Downsample
    X_full = X[:n_train + n_test].reshape(-1, 28, 28)
    factor = 28 // size
    X_down = np.zeros((n_train + n_test, size, size))
    for i in range(size):
        for j in range(size):
            X_down[:, i, j] = X_full[:,
                i*factor:(i+1)*factor,
                j*factor:(j+1)*factor
            ].mean(axis=(1, 2))
    X_all = X_down.reshape(n_train + n_test, -1)
    y_all = y[:n_train + n_test]

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_test, y_test = X_all[n_train:], y_all[n_train:]

    # Scale to [-1, 1]
    X_train_scaled = 2 * X_train - 1
    X_test_scaled = 2 * X_test - 1

    n_dim = size * size

    # Train
    print("Training conditional model...")
    engine = ConditionalCriticalityEngine(n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=0.5)

    for epoch in range(100):
        loss = engine.train_epoch(X_train_scaled, y_train, lr_b=0.02, lr_W=0.005, sigma=0.3)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.4f}")

    # === Test classification on clean images ===
    print("\n=== Classification on Clean Test Images ===")

    correct_energy = 0
    correct_grad = 0

    for i in range(n_test):
        x = X_test_scaled[i]
        true_label = y_test[i]

        pred_energy, _ = classify_by_energy(engine, x)
        pred_grad, _ = classify_by_gradient(engine, x)

        if pred_energy == true_label:
            correct_energy += 1
        if pred_grad == true_label:
            correct_grad += 1

    print(f"  Energy-based accuracy: {correct_energy/n_test*100:.1f}%")
    print(f"  Gradient-based accuracy: {correct_grad/n_test*100:.1f}%")

    # === Test on noisy images ===
    print("\n=== Classification on Noisy Test Images (σ=0.3) ===")

    correct_energy_noisy = 0
    correct_grad_noisy = 0

    for i in range(n_test):
        x = X_test_scaled[i] + np.random.randn(n_dim) * 0.3
        true_label = y_test[i]

        pred_energy, _ = classify_by_energy(engine, x)
        pred_grad, _ = classify_by_gradient(engine, x)

        if pred_energy == true_label:
            correct_energy_noisy += 1
        if pred_grad == true_label:
            correct_grad_noisy += 1

    print(f"  Energy-based accuracy: {correct_energy_noisy/n_test*100:.1f}%")
    print(f"  Gradient-based accuracy: {correct_grad_noisy/n_test*100:.1f}%")

    # === Visualize energy landscape for a few examples ===
    print("\n=== Visualizing Energy Landscape ===")

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for idx, digit in enumerate(range(10)):
        # Find a test example of this digit
        example_idx = np.where(y_test == digit)[0][0]
        x = X_test_scaled[example_idx]

        _, energies = classify_by_energy(engine, x)
        energies = np.array(energies)
        energies = energies - energies.min()  # Normalize

        ax = axes[idx // 5, idx % 5]
        bars = ax.bar(range(10), energies, color=['green' if k == digit else 'gray' for k in range(10)])
        ax.set_title(f'True: {digit}')
        ax.set_xlabel('Class')
        ax.set_ylabel('Energy (relative)')
        ax.set_xticks(range(10))

    plt.suptitle('Energy V_k(x) for Each Class (green = true class, should be lowest)', fontsize=12)
    plt.tight_layout()
    plt.savefig('classification_energy.png', dpi=150)
    print("Saved classification_energy.png")

    # === Full pipeline demo ===
    print("\n=== Full Pipeline Demo ===")

    fig, axes = plt.subplots(4, 10, figsize=(15, 6))

    for digit in range(10):
        # Get a test example
        example_idx = np.where(y_test == digit)[0][0]
        x_clean = X_test_scaled[example_idx]
        x_noisy = x_clean + np.random.randn(n_dim) * 0.5

        # Classify
        pred_class, _ = classify_by_energy(engine, x_noisy)

        # Denoise with predicted class
        x_denoised = engine.denoise(x_noisy.reshape(1, -1), class_idx=pred_class, n_steps=200)[0]

        # Plot
        axes[0, digit].imshow(((x_clean + 1) / 2).reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[0, digit].set_title(f'True: {digit}')
        axes[0, digit].axis('off')

        axes[1, digit].imshow(((x_noisy + 1) / 2).reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[1, digit].axis('off')

        axes[2, digit].set_title(f'Pred: {pred_class}', color='green' if pred_class == digit else 'red')
        axes[2, digit].imshow(((x_denoised + 1) / 2).reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[2, digit].axis('off')

        # Also show what happens with true class
        x_denoised_true = engine.denoise(x_noisy.reshape(1, -1), class_idx=digit, n_steps=200)[0]
        axes[3, digit].imshow(((x_denoised_true + 1) / 2).reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[3, digit].axis('off')

    axes[0, 0].set_ylabel('Clean', rotation=0, ha='right')
    axes[1, 0].set_ylabel('Noisy', rotation=0, ha='right')
    axes[2, 0].set_ylabel('Pred k*', rotation=0, ha='right')
    axes[3, 0].set_ylabel('True k', rotation=0, ha='right')

    plt.suptitle('Full Pipeline: Input → Classify → Denoise (Pred vs True class)', fontsize=12)
    plt.tight_layout()
    plt.savefig('full_pipeline.png', dpi=150)
    print("Saved full_pipeline.png")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
