"""
Test contrastive training for class biases.

Key insight from our derivation:
- Contrastive loss doesn't affect J (cancels out)
- Only affects b_k via Hebbian/anti-Hebbian rule

We need to BALANCE:
1. Score matching (reconstruction) - keeps biases as attractor templates
2. Contrastive (discrimination) - separates class energies

Combined loss: L = L_score + lambda * L_contrastive
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mnist_conditional import ConditionalCriticalityEngine, load_mnist


def compute_energy(engine, x, class_idx):
    """Compute V_k(x) for a single sample."""
    V = (engine.J2 * np.sum(x**2) +
         engine.J4 * np.sum(x**4) +
         np.dot(engine.b[class_idx], x) +
         0.5 * x @ (engine.W @ engine.W.T) @ x)
    return V


def softmax_classification(engine, x, temperature=1.0):
    """Compute p(k|x) = softmax(-V_k(x)/T)"""
    energies = np.array([compute_energy(engine, x, k) for k in range(10)])
    # Subtract max for numerical stability
    energies = energies - energies.max()
    probs = np.exp(-energies / temperature)
    return probs / probs.sum()


def train_combined(engine, X, y, n_dim, n_epochs=100, lr_b=0.02, lr_W=0.005,
                   sigma=0.3, contrastive_weight=0.01, use_crossentropy=False):
    """
    Combined training:
    - Score matching updates both W and b
    - Contrastive/CrossEntropy only updates b
    """
    n_samples = len(X)
    K = engine.n_classes

    for epoch in range(n_epochs):
        # Score matching step (as before)
        score_loss = engine.train_epoch(X, y, lr_b=lr_b, lr_W=lr_W, sigma=sigma)

        # Classification step on biases only
        if contrastive_weight > 0:
            perm = np.random.permutation(n_samples)
            batch_size = 100

            for b_idx in range(n_samples // batch_size):
                batch_indices = perm[b_idx*batch_size:(b_idx+1)*batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                grad_b = np.zeros_like(engine.b)

                for i in range(len(X_batch)):
                    x = X_batch[i]
                    k = int(y_batch[i])

                    if use_crossentropy:
                        # Cross-entropy: L = -log p(k|x) = V_k + log sum_j exp(-V_j)
                        # Gradient: dL/db_k = x * (1 - p(k|x))
                        #           dL/db_j = x * (-p(j|x))  for j != k
                        probs = softmax_classification(engine, x, temperature=1.0)
                        for j in range(K):
                            if j == k:
                                grad_b[j] += x * (1 - probs[j])
                            else:
                                grad_b[j] += x * (-probs[j])
                    else:
                        # Simple contrastive (margin-based)
                        grad_b[k] += x
                        for j in range(K):
                            if j != k:
                                grad_b[j] -= x / (K - 1)

                grad_b /= len(X_batch)
                engine.b -= contrastive_weight * lr_b * grad_b

        if epoch % 20 == 0:
            acc = test_classification(engine, X[:1000], y[:1000])
            print(f"  Epoch {epoch}: score_loss={score_loss:.1f}, acc={acc:.1f}%")

    return engine


def test_classification(engine, X, y):
    """Test energy-based classification: k* = argmin_k V_k(x)"""
    correct = 0
    for i in range(len(X)):
        x = X[i]
        true_label = int(y[i])

        energies = [compute_energy(engine, x, k) for k in range(10)]
        pred = np.argmin(energies)

        if pred == true_label:
            correct += 1

    return 100.0 * correct / len(X)


def unconditional_generation(engine, n_dim, n_samples=10, n_steps=300, dt=0.02):
    """
    Unconditional generation:
    1. Sample class k uniformly
    2. Initialize x randomly
    3. Evolve under V_k
    """
    generated = []
    classes = []

    for i in range(n_samples):
        k = np.random.randint(0, 10)
        classes.append(k)

        x = np.random.randn(n_dim) * 0.5

        for step in range(n_steps):
            grad = engine.grad_V(x.reshape(1, -1), class_idx=k)[0]
            noise = np.sqrt(2 * engine.kT * dt * 0.1) * np.random.randn(n_dim)
            x = x - grad * dt + noise

        generated.append(x)

    return np.array(generated), np.array(classes)


def main():
    print("Loading MNIST...")
    X, y = load_mnist()

    size = 14
    n_train = 5000
    n_dim = size * size

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
    X_scaled = 2 * X_train - 1

    # === Experiment 1: Score matching only (baseline) ===
    print("\n=== Baseline: Score Matching Only ===")
    engine_sm = ConditionalCriticalityEngine(n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=0.5)
    engine_sm = train_combined(engine_sm, X_scaled, y_train, n_dim,
                               n_epochs=100, contrastive_weight=0.0)
    acc_sm = test_classification(engine_sm, X_scaled[:1000], y_train[:1000])

    # === Experiment 2: Cross-entropy training ===
    print("\n=== Combined: Score Matching + Cross-Entropy ===")
    engine_ce = ConditionalCriticalityEngine(n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=0.5)
    engine_ce = train_combined(engine_ce, X_scaled, y_train, n_dim,
                               n_epochs=100, contrastive_weight=1.0, use_crossentropy=True)
    acc_ce = test_classification(engine_ce, X_scaled[:1000], y_train[:1000])

    # === Experiment 3: Stronger cross-entropy ===
    print("\n=== Stronger Cross-Entropy ===")
    engine_ce2 = ConditionalCriticalityEngine(n_dim, n_classes=10, J2=-1.0, J4=0.5, kT=0.5)
    engine_ce2 = train_combined(engine_ce2, X_scaled, y_train, n_dim,
                                n_epochs=100, contrastive_weight=5.0, use_crossentropy=True)
    acc_ce2 = test_classification(engine_ce2, X_scaled[:1000], y_train[:1000])

    # Use cross-entropy model for comparisons
    engine_comb = engine_ce
    acc_comb = acc_ce
    acc_high = acc_ce2

    # === Unconditional generation with best model ===
    print("\n=== Unconditional Generation ===")

    # Score matching model
    gen_sm, cls_sm = unconditional_generation(engine_sm, n_dim, n_samples=20)

    # Combined model
    gen_comb, cls_comb = unconditional_generation(engine_comb, n_dim, n_samples=20)

    # Plot comparison
    fig, axes = plt.subplots(4, 10, figsize=(15, 6))

    for i in range(10):
        # Row 1-2: Score matching generation
        img = ((gen_sm[i] + 1) / 2).reshape(size, size)
        axes[0, i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'k={cls_sm[i]}', fontsize=9)
        axes[0, i].axis('off')

        img = ((gen_sm[i+10] + 1) / 2).reshape(size, size)
        axes[1, i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'k={cls_sm[i+10]}', fontsize=9)
        axes[1, i].axis('off')

        # Row 3-4: Combined model generation
        img = ((gen_comb[i] + 1) / 2).reshape(size, size)
        axes[2, i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title(f'k={cls_comb[i]}', fontsize=9)
        axes[2, i].axis('off')

        img = ((gen_comb[i+10] + 1) / 2).reshape(size, size)
        axes[3, i].imshow(np.clip(img, 0, 1), cmap='gray', vmin=0, vmax=1)
        axes[3, i].set_title(f'k={cls_comb[i+10]}', fontsize=9)
        axes[3, i].axis('off')

    axes[0, 0].set_ylabel('Score\nMatching', rotation=0, ha='right', fontsize=10, labelpad=30)
    axes[2, 0].set_ylabel('Combined\nTraining', rotation=0, ha='right', fontsize=10, labelpad=30)

    plt.suptitle('Unconditional Generation: Sample k uniformly, then generate x|k', fontsize=12)
    plt.tight_layout()
    plt.savefig('unconditional_generation.png', dpi=150, bbox_inches='tight')
    print("Saved unconditional_generation.png")

    # === Bias comparison ===
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))

    for k in range(10):
        bias_sm = (-engine_sm.b[k]).reshape(size, size)
        axes[0, k].imshow(bias_sm, cmap='RdBu_r',
                         vmin=-np.abs(bias_sm).max(), vmax=np.abs(bias_sm).max())
        axes[0, k].set_title(str(k), fontsize=10)
        axes[0, k].axis('off')

        bias_comb = (-engine_comb.b[k]).reshape(size, size)
        axes[1, k].imshow(bias_comb, cmap='RdBu_r',
                         vmin=-np.abs(bias_comb).max(), vmax=np.abs(bias_comb).max())
        axes[1, k].axis('off')

        diff = bias_comb - bias_sm
        axes[2, k].imshow(diff, cmap='RdBu_r',
                         vmin=-np.abs(diff).max(), vmax=np.abs(diff).max())
        axes[2, k].axis('off')

    axes[0, 0].set_ylabel('Score\nMatching', rotation=0, ha='right', fontsize=10, labelpad=30)
    axes[1, 0].set_ylabel('Combined', rotation=0, ha='right', fontsize=10, labelpad=30)
    axes[2, 0].set_ylabel('Difference', rotation=0, ha='right', fontsize=10, labelpad=30)

    plt.suptitle('Bias Templates: Score Matching vs Combined Training', fontsize=12)
    plt.tight_layout()
    plt.savefig('bias_comparison_training.png', dpi=150, bbox_inches='tight')
    print("Saved bias_comparison_training.png")

    # === Summary ===
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Classification Accuracy:")
    print(f"  Score matching only:        {acc_sm:.1f}%")
    print(f"  + Cross-entropy (1.0):      {acc_comb:.1f}%")
    print(f"  + Cross-entropy (5.0):      {acc_high:.1f}%")
    print(f"\nRandom baseline: 10%")
    print("="*60)


if __name__ == "__main__":
    main()
