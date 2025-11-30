"""
Visualize learned class biases and compare with digit averages.
Shows the Hebbian/anti-Hebbian structure from contrastive interpretation.
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
    X_scaled = 2 * X_train - 1  # Scale to [-1, 1]

    n_dim = size * size

    # Compute real digit averages
    digit_avgs = np.zeros((10, n_dim))
    for k in range(10):
        mask = y_train == k
        digit_avgs[k] = X_scaled[mask].mean(axis=0)

    print("Training model...")
    engine = ConditionalCriticalityEngine(n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=0.5)

    for epoch in range(100):
        loss = engine.train_epoch(X_scaled, y_train, lr_b=0.02, lr_W=0.005, sigma=0.3)
        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.4f}")

    # === Figure 1: Learned biases as templates ===
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    for k in range(10):
        ax = axes[k // 5, k % 5]
        # Negative bias shows what the class "wants" (pulls toward +1)
        bias_img = (-engine.b[k]).reshape(size, size)
        im = ax.imshow(bias_img, cmap='RdBu_r', vmin=-np.abs(bias_img).max(), vmax=np.abs(bias_img).max())
        ax.set_title(f'$-b_{k}$', fontsize=12)
        ax.axis('off')

    plt.suptitle('Learned Class Biases (Attractor Templates)', fontsize=14)
    plt.tight_layout()
    plt.savefig('class_biases.png', dpi=150, bbox_inches='tight')
    print("Saved class_biases.png")

    # === Figure 2: Comparison - Real averages vs Learned biases ===
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))

    for k in range(10):
        # Row 1: Real digit averages
        avg_img = ((digit_avgs[k] + 1) / 2).reshape(size, size)
        axes[0, k].imshow(np.clip(avg_img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, k].set_title(str(k), fontsize=11)
        axes[0, k].axis('off')

        # Row 2: Learned biases (-b_k normalized to [0,1] for visualization)
        bias = -engine.b[k]
        bias_norm = (bias - bias.min()) / (bias.max() - bias.min() + 1e-8)
        axes[1, k].imshow(bias_norm.reshape(size, size), cmap='gray', vmin=0, vmax=1)
        axes[1, k].axis('off')

        # Row 3: Correlation map (where bias matches average)
        corr = digit_avgs[k] * (-engine.b[k])  # Positive where they agree
        axes[2, k].imshow(corr.reshape(size, size), cmap='RdBu_r',
                         vmin=-np.abs(corr).max(), vmax=np.abs(corr).max())
        axes[2, k].axis('off')

    axes[0, 0].set_ylabel('Data\nAverage', rotation=0, ha='right', fontsize=10, labelpad=30)
    axes[1, 0].set_ylabel('Learned\nBias $-b_k$', rotation=0, ha='right', fontsize=10, labelpad=30)
    axes[2, 0].set_ylabel('Agreement\n$\\bar{x}_k \\cdot (-b_k)$', rotation=0, ha='right', fontsize=10, labelpad=30)

    plt.suptitle('Comparison: Data Averages vs Learned Biases', fontsize=12)
    plt.tight_layout()
    plt.savefig('bias_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved bias_comparison.png")

    # === Figure 3: Contrastive learning interpretation ===
    # Show how biases create class separation
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))

    # For each class, show: this class's bias vs average of other biases
    for k in range(10):
        ax = axes[k // 5, k % 5]

        # This class bias
        b_k = -engine.b[k]

        # Average of other class biases
        b_others = np.mean([-engine.b[j] for j in range(10) if j != k], axis=0)

        # Difference: what makes this class unique
        diff = b_k - b_others

        im = ax.imshow(diff.reshape(size, size), cmap='RdBu_r',
                      vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        ax.set_title(f'Digit {k}', fontsize=11)
        ax.axis('off')

    plt.suptitle('Contrastive Signal: $-b_k + \\frac{1}{K-1}\\sum_{j\\neq k} b_j$\n(What makes each digit unique)', fontsize=12)
    plt.tight_layout()
    plt.savefig('contrastive_biases.png', dpi=150, bbox_inches='tight')
    print("Saved contrastive_biases.png")

    # === Figure 4: Energy landscape for classification ===
    # Show V_k(x) for a test digit across all classes
    fig, axes = plt.subplots(2, 5, figsize=(14, 5))

    # Pick one example of each digit
    test_examples = []
    for k in range(10):
        idx = np.where(y_train == k)[0][0]
        test_examples.append(X_scaled[idx])

    # For digit "3", show energy under each V_k
    test_digit = 3
    x_test = test_examples[test_digit]

    energies = []
    for k in range(10):
        # Compute V_k(x)
        V_k = (engine.J2 * np.sum(x_test**2) +
               engine.J4 * np.sum(x_test**4) +
               np.dot(engine.b[k], x_test) +
               0.5 * x_test @ (engine.W @ engine.W.T) @ x_test)
        energies.append(V_k)

    energies = np.array(energies)

    # Bar plot of energies
    ax_main = fig.add_subplot(1, 2, 1)
    colors = ['green' if k == test_digit else 'red' for k in range(10)]
    bars = ax_main.bar(range(10), energies, color=colors, alpha=0.7)
    ax_main.set_xlabel('Class k', fontsize=11)
    ax_main.set_ylabel('$V_k(x)$', fontsize=11)
    ax_main.set_title(f'Energy of digit {test_digit} under each $V_k$', fontsize=12)
    ax_main.set_xticks(range(10))
    ax_main.axhline(y=energies[test_digit], color='green', linestyle='--', alpha=0.5)

    # Show the test digit
    ax_img = fig.add_subplot(1, 2, 2)
    ax_img.imshow(((x_test + 1)/2).reshape(size, size), cmap='gray', vmin=0, vmax=1)
    ax_img.set_title(f'Test digit: {test_digit}', fontsize=12)
    ax_img.axis('off')

    # Add text showing classification
    pred = np.argmin(energies)
    ax_main.annotate(f'argmin = {pred}', xy=(pred, energies[pred]),
                    xytext=(pred+0.5, energies[pred]-10),
                    fontsize=10, color='blue',
                    arrowprops=dict(arrowstyle='->', color='blue'))

    plt.tight_layout()
    plt.savefig('energy_classification.png', dpi=150, bbox_inches='tight')
    print("Saved energy_classification.png")

    # Print classification accuracy
    print("\n=== Classification Test ===")
    correct = 0
    n_test = min(1000, len(X_scaled))
    for i in range(n_test):
        x = X_scaled[i]
        true_label = y_train[i]

        energies = []
        for k in range(10):
            V_k = (engine.J2 * np.sum(x**2) +
                   engine.J4 * np.sum(x**4) +
                   np.dot(engine.b[k], x) +
                   0.5 * x @ (engine.W @ engine.W.T) @ x)
            energies.append(V_k)

        pred = np.argmin(energies)
        if pred == true_label:
            correct += 1

    acc = correct / n_test * 100
    print(f"Classification accuracy (score matching only): {acc:.1f}%")
    print("(Low because biases optimized for reconstruction, not discrimination)")

    print("\nDone!")


if __name__ == "__main__":
    main()
