"""
Redo trajectory figure: start from VERY noisy 3, show particles being attracted.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mnist_conditional import ConditionalCriticalityEngine, load_mnist


def main():
    X, y = load_mnist()

    size = 14
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
    X_scaled = 2 * X_train - 1

    n_dim = size * size

    print("Training model...")
    engine = ConditionalCriticalityEngine(n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=0.5)

    for epoch in range(100):
        loss = engine.train_epoch(X_scaled, y_train, lr_b=0.02, lr_W=0.005, sigma=0.3)
        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.4f}")

    # Get a clean 3
    idx_3 = np.where(y_train == 3)[0][0]
    x_clean = X_scaled[idx_3]

    # Start with HEAVY noise - barely recognizable
    np.random.seed(42)
    noise_level = 0.8  # Much more noise!
    x_noisy = x_clean + np.random.randn(n_dim) * noise_level

    # Also add salt-and-pepper noise
    salt_pepper = np.random.rand(n_dim)
    x_noisy[salt_pepper < 0.1] = 1.0   # salt
    x_noisy[salt_pepper > 0.9] = -1.0  # pepper

    print(f"\nRelaxing from heavy noise (σ={noise_level} + salt/pepper)...")

    # Run relaxation with class-conditional potential
    x = x_noisy.copy().reshape(1, -1)
    dt = 0.02
    n_steps = 300

    # Collect trajectory at specific steps
    steps_to_show = [0, 3, 7, 15, 30, 60, 120, 200, 300]
    trajectory = {0: x_noisy.copy()}

    for step in range(1, n_steps + 1):
        drift = engine.drift(x, class_idx=3)
        # Low noise during relaxation - follow the attractor
        x = x + drift * dt + np.sqrt(2 * engine.kT * dt * 0.05) * np.random.randn(*x.shape)

        if step in steps_to_show:
            trajectory[step] = x[0].copy()

    # Plot
    fig, axes = plt.subplots(1, len(steps_to_show) + 1, figsize=(18, 2.2))

    # Original clean 3
    img = ((x_clean + 1) / 2).reshape(size, size)
    axes[0].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Target', fontsize=11)
    axes[0].axis('off')

    # Trajectory
    for i, step in enumerate(steps_to_show):
        img = ((trajectory[step] + 1) / 2).reshape(size, size)
        axes[i + 1].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[i + 1].set_title(f't={step}', fontsize=11)
        axes[i + 1].axis('off')

    plt.suptitle('Criticality Engine: Particles Attracted to Digit 3 Attractor Basin', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('criticality_trajectory.png', dpi=150, bbox_inches='tight')
    print("Saved criticality_trajectory.png")

    # Also make a version showing the "energy landscape" interpretation
    fig, axes = plt.subplots(2, len(steps_to_show) + 1, figsize=(18, 4.5))

    # Top row: images
    img = ((x_clean + 1) / 2).reshape(size, size)
    axes[0, 0].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Target', fontsize=10)
    axes[0, 0].axis('off')

    for i, step in enumerate(steps_to_show):
        img = ((trajectory[step] + 1) / 2).reshape(size, size)
        axes[0, i + 1].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, i + 1].set_title(f't={step}', fontsize=10)
        axes[0, i + 1].axis('off')

    # Bottom row: gradient magnitude (how far from equilibrium)
    axes[1, 0].axis('off')

    grad_mags = []
    for i, step in enumerate(steps_to_show):
        grad = engine.grad_V(trajectory[step], class_idx=3)
        grad_mag = np.sqrt(np.sum(grad**2))
        grad_mags.append(grad_mag)

        # Show gradient as heatmap
        grad_img = np.abs(grad).reshape(size, size)
        axes[1, i + 1].imshow(grad_img, cmap='hot', vmin=0, vmax=grad_img.max() * 0.5)
        axes[1, i + 1].set_title(f'|∇V|={grad_mag:.0f}', fontsize=9)
        axes[1, i + 1].axis('off')

    axes[0, 0].set_ylabel('State x(t)', fontsize=10)
    axes[1, 0].set_ylabel('Gradient |∇V|', fontsize=10)
    axes[1, 0].text(0.5, 0.5, '(force toward\nattractor)', ha='center', va='center', fontsize=9, transform=axes[1, 0].transAxes)

    plt.suptitle('Relaxation Dynamics: Gradient Decreases as System Approaches Attractor', fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig('trajectory_with_gradient.png', dpi=150, bbox_inches='tight')
    print("Saved trajectory_with_gradient.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
