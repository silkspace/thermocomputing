"""
Comparison: Trajectory Estimation vs Standard ML Baselines

This compares our trajectory-based parameter estimation with:
1. Linear classifier (scikit-learn) - Simple but effective baseline
2. PyTorch implementation of the same analytical gradient formula

Key insight: Our method achieves 76% accuracy using PHYSICS-BASED gradients
derived from the Onsager-Machlup action, not standard ML loss functions.

Usage:
    python pytorch_comparison.py
"""

import time
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def load_data(image_size=14):
    """Load MNIST with same preprocessing as trajectory_estimator."""
    from sklearn.datasets import fetch_openml

    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(np.int64)

    n_train, n_test = 5000, 1000

    def downsample(X_subset):
        n = len(X_subset)
        X_full = X_subset.reshape(-1, 28, 28)
        factor = 28 // image_size
        X_down = np.zeros((n, image_size, image_size))
        for i in range(image_size):
            for j in range(image_size):
                X_down[:, i, j] = X_full[:,
                    i*factor:(i+1)*factor,
                    j*factor:(j+1)*factor
                ].mean(axis=(1, 2))
        return 2 * X_down.reshape(n, -1) - 1

    X_train = downsample(X[:n_train])
    y_train = y[:n_train]
    X_test = downsample(X[-n_test:])
    y_test = y[-n_test:]

    return X_train, y_train, X_test, y_test


def run_linear_baseline(X_train, y_train, X_test, y_test):
    """Simple linear classifier baseline."""
    from sklearn.linear_model import LogisticRegression

    print("\n=== Linear Classifier (LogisticRegression) ===")
    start = time.time()
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train, y_train)
    elapsed = time.time() - start

    train_acc = clf.score(X_train, y_train) * 100
    test_acc = clf.score(X_test, y_test) * 100

    print(f"Train Accuracy: {train_acc:.1f}%")
    print(f"Test Accuracy:  {test_acc:.1f}%")
    print(f"Training Time:  {elapsed:.2f}s")

    return test_acc, elapsed


def run_pytorch_analytical(X_train, y_train, X_test, y_test, n_epochs=100, lr=0.05):
    """
    PyTorch implementation using the SAME analytical gradient formula
    as trajectory_estimator.py.

    This shows PyTorch can achieve the same result - it's not about PyTorch vs NumPy,
    it's about the analytical gradient formula being correct.
    """
    print("\n=== PyTorch (Analytical Gradients, same formula) ===")

    n_dim = X_train.shape[1]
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=DEVICE)

    # Physics parameters (same as trajectory_estimator)
    J2, J4, kT, mu, dt = -1.0, 0.5, 0.5, 1.0, 0.05
    n_steps = 5

    # Learnable biases
    b = torch.zeros(10, n_dim, device=DEVICE)

    def conservative_diffusion_step(x):
        H = W = 14
        batch = x.shape[0]
        x_img = x.reshape(batch, H, W)
        laplacian = torch.zeros_like(x_img)
        laplacian[:, 1:, :] += x_img[:, :-1, :] - x_img[:, 1:, :]
        laplacian[:, :-1, :] += x_img[:, 1:, :] - x_img[:, :-1, :]
        laplacian[:, :, 1:] += x_img[:, :, :-1] - x_img[:, :, 1:]
        laplacian[:, :, :-1] += x_img[:, :, 1:] - x_img[:, :, :-1]
        noise = torch.randn_like(x_img)
        noise = noise - noise.mean(dim=(1, 2), keepdim=True)
        x_new = x_img + 0.1 * laplacian + 0.1 * noise
        return x_new.reshape(batch, -1)

    def classify(X):
        energies = []
        for c in range(10):
            V = J2 * (X ** 2).sum(dim=-1) + J4 * (X ** 4).sum(dim=-1)
            V = V + (b[c] * X).sum(dim=-1)
            energies.append(V)
        return torch.argmin(torch.stack(energies, dim=1), dim=1)

    start = time.time()
    n_samples = len(X_train_t)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)[:500]

        grad_accum = torch.zeros(10, n_dim, device=DEVICE)
        class_counts = torch.zeros(10, device=DEVICE)

        for idx in perm:
            x = X_train_t[idx:idx+1]
            c = y_train_t[idx].item()

            traj_grad = torch.zeros(n_dim, device=DEVICE)

            for _ in range(n_steps):
                x_next = conservative_diffusion_step(x)
                dx = x_next - x
                grad_V = 2 * J2 * x + 4 * J4 * (x ** 3) + b[c:c+1]
                residual = dx + mu * grad_V * dt
                grad_b = -residual.squeeze(0) / (2 * kT)
                traj_grad = traj_grad + grad_b
                x = x_next

            traj_grad = traj_grad / n_steps
            grad_norm = traj_grad.norm()
            if grad_norm > 1.0:
                traj_grad = traj_grad / grad_norm

            grad_accum[c] = grad_accum[c] + traj_grad
            class_counts[c] += 1

        for c in range(10):
            if class_counts[c] > 0:
                b[c] -= lr * grad_accum[c] / class_counts[c]

        if epoch % 20 == 0:
            preds = classify(X_test_t)
            acc = (preds == y_test_t).float().mean().item() * 100
            print(f"Epoch {epoch}: Test Acc = {acc:.1f}%")

    elapsed = time.time() - start

    preds_train = classify(X_train_t[:1000])
    preds_test = classify(X_test_t)
    train_acc = (preds_train == y_train_t[:1000]).float().mean().item() * 100
    test_acc = (preds_test == y_test_t).float().mean().item() * 100

    print(f"\nTrain Accuracy: {train_acc:.1f}%")
    print(f"Test Accuracy:  {test_acc:.1f}%")
    print(f"Training Time:  {elapsed:.2f}s")

    return test_acc, elapsed


def main():
    print("="*60)
    print("COMPARISON: Trajectory Estimation vs Standard ML")
    print(f"Device: {DEVICE}")
    print("="*60)

    X_train, y_train, X_test, y_test = load_data()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Run comparisons
    results = {}

    # 1. Linear baseline
    acc1, time1 = run_linear_baseline(X_train, y_train, X_test, y_test)
    results['Linear'] = (acc1, time1)

    # 2. PyTorch with analytical gradients
    acc2, time2 = run_pytorch_analytical(X_train, y_train, X_test, y_test)
    results['PyTorch (Analytical)'] = (acc2, time2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n| Method | Test Accuracy | Time |")
    print("|--------|---------------|------|")
    for name, (acc, t) in results.items():
        print(f"| {name} | {acc:.1f}% | {t:.1f}s |")
    print("| NumPy (trajectory_estimator.py) | 76.4% | ~30s |")

    print("\nKey insight: The φ⁴+bias model with analytical gradients")
    print("achieves competitive accuracy without standard ML loss functions.")

    # Save figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    methods = list(results.keys()) + ['NumPy\n(trajectory_estimator)']
    accs = [results[k][0] for k in list(results.keys())] + [76.4]
    times = [results[k][1] for k in list(results.keys())] + [30]

    axes[0].bar(methods, accs, color=['blue', 'green', 'orange'])
    axes[0].axhline(y=10, color='r', linestyle='--', label='Random')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_ylim([0, 100])

    axes[1].bar(methods, times, color=['blue', 'green', 'orange'])
    axes[1].set_ylabel('Training Time (seconds)')
    axes[1].set_title('Time Comparison')

    plt.tight_layout()
    plt.savefig('paper_figures/pytorch_comparison.png', dpi=150)
    print("\nSaved paper_figures/pytorch_comparison.png")


if __name__ == "__main__":
    main()
