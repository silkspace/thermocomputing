"""
Visualize digit attractors: Start from blank/random, turn on each digit's potential,
watch where pixels fall.

This is the key demonstration: the learned b_k creates an attractor basin
that organizes random noise into digit structure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mnist_conditional import ConditionalCriticalityEngine, load_mnist


def main():
    # Load trained model
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

    # Train the model
    print("Training conditional model...")
    engine = ConditionalCriticalityEngine(n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=0.5)

    for epoch in range(100):
        loss = engine.train_epoch(X_scaled, y_train, lr_b=0.02, lr_W=0.005, sigma=0.3)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.4f}")

    # === ATTRACTOR VISUALIZATION ===
    print("\n=== Visualizing Attractors ===")

    # For each digit, start from gray (x=0) and relax under that digit's potential
    n_steps = 300
    dt = 0.02

    # Store trajectories
    trajectories = {}
    steps_to_show = [0, 10, 30, 50, 100, 150, 200, 300]

    for digit in range(10):
        print(f"  Relaxing under digit {digit} potential...")

        # Start from gray (neutral, x=0)
        x = np.zeros((1, n_dim))

        traj = [x.copy()]
        for step in range(n_steps):
            # Low noise - mostly follow the drift
            drift = engine.drift(x, class_idx=digit)
            x = x + drift * dt + np.sqrt(2 * engine.kT * dt * 0.1) * np.random.randn(*x.shape)
            if step + 1 in steps_to_show:
                traj.append(x.copy())

        trajectories[digit] = traj

    # Plot: 10 digits × trajectory steps
    fig, axes = plt.subplots(10, len(steps_to_show), figsize=(16, 20))

    for digit in range(10):
        traj = trajectories[digit]
        for i, step in enumerate(steps_to_show):
            idx = steps_to_show.index(step)
            if idx < len(traj):
                img = ((traj[idx][0] + 1) / 2).reshape(size, size)
                axes[digit, i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
            axes[digit, i].axis('off')
            if digit == 0:
                axes[digit, i].set_title(f't={step}', fontsize=10)
        axes[digit, 0].set_ylabel(f'{digit}', rotation=0, ha='right', fontsize=14, fontweight='bold')

    plt.suptitle('Attractor Dynamics: Gray → Digit (each row = one digit potential)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('attractors_from_gray.png', dpi=150, bbox_inches='tight')
    print("Saved attractors_from_gray.png")

    # === FROM RANDOM NOISE ===
    print("\n=== From Random Noise ===")

    trajectories_noise = {}
    np.random.seed(42)  # Reproducible

    for digit in range(10):
        # Start from random noise
        x = np.random.randn(1, n_dim) * 0.5

        traj = [x.copy()]
        for step in range(n_steps):
            drift = engine.drift(x, class_idx=digit)
            x = x + drift * dt + np.sqrt(2 * engine.kT * dt * 0.1) * np.random.randn(*x.shape)
            if step + 1 in steps_to_show:
                traj.append(x.copy())

        trajectories_noise[digit] = traj

    fig, axes = plt.subplots(10, len(steps_to_show), figsize=(16, 20))

    for digit in range(10):
        traj = trajectories_noise[digit]
        for i, step in enumerate(steps_to_show):
            idx = steps_to_show.index(step)
            if idx < len(traj):
                img = ((traj[idx][0] + 1) / 2).reshape(size, size)
                axes[digit, i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
            axes[digit, i].axis('off')
            if digit == 0:
                axes[digit, i].set_title(f't={step}', fontsize=10)
        axes[digit, 0].set_ylabel(f'{digit}', rotation=0, ha='right', fontsize=14, fontweight='bold')

    plt.suptitle('Attractor Dynamics: Random Noise → Digit', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('attractors_from_noise.png', dpi=150, bbox_inches='tight')
    print("Saved attractors_from_noise.png")

    # === FINAL ATTRACTORS (clean view) ===
    print("\n=== Final Attractors ===")

    fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))

    for digit in range(10):
        # From gray
        final_gray = trajectories[digit][-1][0]
        img_gray = ((final_gray + 1) / 2).reshape(size, size)
        axes[0, digit].imshow(np.clip(img_gray, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, digit].set_title(str(digit), fontsize=12, fontweight='bold')
        axes[0, digit].axis('off')

        # From noise
        final_noise = trajectories_noise[digit][-1][0]
        img_noise = ((final_noise + 1) / 2).reshape(size, size)
        axes[1, digit].imshow(np.clip(img_noise, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, digit].axis('off')

    axes[0, 0].set_ylabel('From\ngray', rotation=0, ha='right', fontsize=10)
    axes[1, 0].set_ylabel('From\nnoise', rotation=0, ha='right', fontsize=10)

    plt.suptitle('Learned Attractors: Where Pixels Fall Under Each Digit Potential', fontsize=12)
    plt.tight_layout()
    plt.savefig('attractors_final.png', dpi=150)
    print("Saved attractors_final.png")

    # === COMPARE TO REAL DIGIT AVERAGES ===
    print("\n=== Compare to Real Digit Averages ===")

    fig, axes = plt.subplots(3, 10, figsize=(15, 5))

    for digit in range(10):
        # Real average
        mask = (y_train == digit)
        avg = X_scaled[mask].mean(0)
        img_avg = ((avg + 1) / 2).reshape(size, size)
        axes[0, digit].imshow(np.clip(img_avg, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, digit].set_title(str(digit), fontsize=12, fontweight='bold')
        axes[0, digit].axis('off')

        # Learned attractor (from gray)
        final = trajectories[digit][-1][0]
        img_final = ((final + 1) / 2).reshape(size, size)
        axes[1, digit].imshow(np.clip(img_final, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, digit].axis('off')

        # Learned bias template
        bias_img = ((-engine.b[digit] + 1) / 2).reshape(size, size)
        axes[2, digit].imshow(bias_img, cmap='gray')
        axes[2, digit].axis('off')

    axes[0, 0].set_ylabel('Real\navg', rotation=0, ha='right', fontsize=10)
    axes[1, 0].set_ylabel('Learned\nattractor', rotation=0, ha='right', fontsize=10)
    axes[2, 0].set_ylabel('Bias\ntemplate', rotation=0, ha='right', fontsize=10)

    plt.suptitle('Comparison: Real Average vs Learned Attractor vs Bias Template', fontsize=12)
    plt.tight_layout()
    plt.savefig('attractors_comparison.png', dpi=150)
    print("Saved attractors_comparison.png")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
