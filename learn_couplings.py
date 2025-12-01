"""
CORRECT Coupling Learning via Thermal Pixel Dynamics

The RIGHT way:
- Pixels are PARTICLES with mass (intensity)
- They EXCHANGE intensity with each other (conserves total)
- If swap doesn't change image → pixels are correlated
- If swap changes image → pixels are uncorrelated

NO ASSUMPTIONS about connectivity. Learn J from observing exchanges.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

Path('paper_figures').mkdir(exist_ok=True)


@dataclass
class Exchange:
    """A single intensity exchange between two pixels."""
    i: int          # Source pixel index
    j: int          # Target pixel index
    delta: float    # Amount transferred (positive = i→j)


@dataclass
class TrajectoryStep:
    """One step in a thermal trajectory."""
    state: np.ndarray           # Pixel values after this step
    exchanges: List[Exchange]   # Exchanges that occurred in this step


@dataclass
class Trajectory:
    """A full thermal trajectory."""
    steps: List[TrajectoryStep]

    @property
    def initial_state(self) -> np.ndarray:
        return self.steps[0].state

    @property
    def final_state(self) -> np.ndarray:
        return self.steps[-1].state

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def total_mass_conserved(self) -> bool:
        """Check if total intensity is conserved."""
        initial = self.initial_state.sum()
        final = self.final_state.sum()
        return np.abs(initial - final) < 1e-6


def load_mnist_raw(n_train: int = 5000, n_test: int = 1000, image_size: int = 14):
    """Load MNIST, downsample, flatten."""
    from sklearn.datasets import fetch_openml
    from skimage.transform import resize

    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data.astype(np.float32), mnist.target.astype(int)
    X = X / 255.0

    n_samples = len(X)
    X_small = np.zeros((n_samples, image_size * image_size), dtype=np.float32)
    for i in range(n_samples):
        img = X[i].reshape(28, 28)
        img_small = resize(img, (image_size, image_size), anti_aliasing=True)
        X_small[i] = img_small.flatten()

    X_train, y_train = X_small[:n_train], y[:n_train]
    X_test, y_test = X_small[-n_test:], y[-n_test:]
    print(f"Train: {n_train}, Test: {n_test}")
    return X_train, y_train, X_test, y_test


class ThermalPixelExchange:
    """
    Correct thermal dynamics: pixels EXCHANGE intensity.

    This is Kawasaki-like dynamics that conserves total mass.
    """

    def __init__(self, n_dim: int, kT: float = 1.0):
        self.n_dim = n_dim
        self.kT = kT

    def step(self, x: np.ndarray, n_exchanges: Optional[int] = None) -> TrajectoryStep:
        """
        One thermal step: multiple random pair exchanges.

        Each exchange:
        1. Pick random pair (i, j)
        2. Transfer some intensity from one to other
        3. Amount is thermal (random, scaled by kT)
        4. Conserves total: x[i] + x[j] stays constant
        """
        if n_exchanges is None:
            n_exchanges = self.n_dim // 4

        x_new = x.copy()
        exchanges: List[Exchange] = []

        for _ in range(n_exchanges):
            # Pick random pair
            i, j = np.random.choice(self.n_dim, 2, replace=False)

            # Amount to transfer: thermal fluctuation
            delta = np.sqrt(self.kT) * np.random.randn()

            # Clamp to valid range (can't go negative)
            delta = np.clip(delta, -x_new[j], x_new[i])

            # Execute exchange
            x_new[i] -= delta
            x_new[j] += delta

            exchanges.append(Exchange(i=int(i), j=int(j), delta=float(delta)))

        return TrajectoryStep(state=x_new, exchanges=exchanges)

    def trajectory(self, x: np.ndarray, n_steps: int,
                   n_exchanges_per_step: Optional[int] = None) -> Trajectory:
        """Generate a trajectory of thermal evolution."""
        steps = [TrajectoryStep(state=x.copy(), exchanges=[])]

        current = x.copy()
        for _ in range(n_steps):
            step = self.step(current, n_exchanges_per_step)
            steps.append(step)
            current = step.state

        return Trajectory(steps=steps)


class CouplingLearner:
    """
    Learn coupling matrix J from observing pixel exchange dynamics.

    The insight:
    - If pixels i,j have similar values, exchanging them doesn't change much
    - If they have different values, exchange is visible
    - J_ij should capture: "how correlated are pixels i and j across the dataset"

    For energy E = 0.5 * x' J x:
    - Exchanging i,j changes E depending on J and x values
    - Learn J so that observed exchanges are "explained" by the energy landscape
    """

    def __init__(self, n_dim: int, n_classes: int, kT: float = 1.0):
        self.n_dim = n_dim
        self.n_classes = n_classes
        self.kT = kT

        # Class-specific coupling matrices (FULL DENSE, no assumptions!)
        self.J = np.zeros((n_classes, n_dim, n_dim))

    def energy(self, x: np.ndarray, class_idx: int) -> float:
        """Energy: E_c(x) = 0.5 * x' J_c x"""
        return 0.5 * x @ self.J[class_idx] @ x

    def energy_change_for_exchange(self, x: np.ndarray, i: int, j: int, delta: float, class_idx: int) -> float:
        """
        Compute energy change for exchanging delta from pixel i to pixel j.

        Before: x_i, x_j
        After: x_i - delta, x_j + delta

        ΔE = E(x') - E(x) where x' is state after exchange
        """
        J = self.J[class_idx]

        # Energy contributions involving i and j
        # E = 0.5 * sum_kl J_kl x_k x_l
        # Changes affect: terms with k=i or k=j or l=i or l=j

        # Direct calculation of ΔE:
        # ΔE = 0.5 * [(x_i - δ)² J_ii + (x_j + δ)² J_jj + 2(x_i - δ)(x_j + δ) J_ij
        #            + 2 sum_{k≠i,j} [(x_i - δ) J_ik x_k + (x_j + δ) J_jk x_k]]
        #    - 0.5 * [x_i² J_ii + x_j² J_jj + 2 x_i x_j J_ij
        #            + 2 sum_{k≠i,j} [x_i J_ik x_k + x_j J_jk x_k]]

        # Simplifying:
        # ΔE = 0.5 * [(-2 x_i δ + δ²) J_ii + (2 x_j δ + δ²) J_jj
        #            + 2 (x_i x_j + x_i δ - x_j δ - δ²) J_ij - 2 x_i x_j J_ij
        #            + 2 sum_{k≠i,j} [-δ J_ik x_k + δ J_jk x_k]]

        # ΔE = 0.5 * [-2 x_i δ J_ii + δ² J_ii + 2 x_j δ J_jj + δ² J_jj
        #            + 2 δ (x_i - x_j - δ) J_ij
        #            + 2 δ sum_{k≠i,j} (J_jk - J_ik) x_k]

        # For small δ (ignore δ² terms):
        # ΔE ≈ δ * [-x_i J_ii + x_j J_jj + (x_i - x_j) J_ij + sum_{k≠i,j} (J_jk - J_ik) x_k]
        #    = δ * [sum_k (J_jk - J_ik) x_k + (J_ij - J_ji)/2 * (x_i - x_j)]  # if J symmetric
        #    = δ * [J_j · x - J_i · x]  where J_i is row i of J
        #    = δ * (J[j, :] - J[i, :]) @ x

        dE = delta * (J[j, :] - J[i, :]) @ x
        return dE

    def compute_exchange_gradient(self, x: np.ndarray, exchange: Exchange,
                                   class_idx: int) -> np.ndarray:
        """
        Compute gradient of J from observing an exchange.

        The key insight: we observe an exchange (i → j, delta).
        We want J such that this exchange is "likely" under the energy landscape.

        Using score matching / likelihood gradient:
        The probability of exchange delta is related to energy change.
        We want to maximize log P(delta | x, J).

        For thermal dynamics: P(delta) ∝ exp(-ΔE / kT)
        So log P ∝ -ΔE / kT
        ∂ log P / ∂ J = -1/kT * ∂(ΔE) / ∂J

        ΔE = delta * (J[j,:] - J[i,:]) @ x
        ∂(ΔE)/∂J[j,k] = delta * x[k]
        ∂(ΔE)/∂J[i,k] = -delta * x[k]
        """
        grad = np.zeros((self.n_dim, self.n_dim))

        # Gradient w.r.t. J[j, k] for all k
        grad[exchange.j, :] = exchange.delta * x / self.kT
        # Gradient w.r.t. J[i, k] for all k
        grad[exchange.i, :] = -exchange.delta * x / self.kT

        # Negative because we want to maximize log likelihood
        return -grad

    def classify(self, x: np.ndarray) -> int:
        """Classify by minimum energy."""
        energies = [self.energy(x, c) for c in range(self.n_classes)]
        return np.argmin(energies)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        correct = sum(1 for i in range(len(X)) if self.classify(X[i]) == y[i])
        return correct / len(X) * 100


@dataclass
class TrainingHistory:
    """Training history with typed fields."""
    epochs: List[int]
    train_acc: List[float]
    test_acc: List[float]
    J_norm: List[float]


def train_from_exchanges(
    learner: CouplingLearner,
    thermalizer: ThermalPixelExchange,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 50,
    n_steps: int = 10,
    n_exchanges_per_step: int = 50,
    lr: float = 0.01,
    samples_per_epoch: int = 500,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None
) -> TrainingHistory:
    """
    Train J by observing thermal exchange trajectories.

    For each data sample:
    1. Apply thermal exchanges (pixels swap intensity)
    2. Observe what exchanges happened
    3. Update J to "explain" the observed exchanges
    """
    n_samples = len(X_train)
    n_dim = X_train.shape[1]
    n_classes = learner.n_classes

    history = TrainingHistory(epochs=[], train_acc=[], test_acc=[], J_norm=[])

    print(f"\nTraining: {n_epochs} epochs, {n_steps} steps/sample, {n_exchanges_per_step} exchanges/step")
    print(f"kT = {thermalizer.kT}, lr = {lr}")
    print(f"Learning {n_classes} J matrices, each {n_dim}x{n_dim} = {n_dim**2:,} params")

    for epoch in range(n_epochs):
        perm = np.random.permutation(n_samples)
        n_batch = min(samples_per_epoch, n_samples)

        # Accumulate gradients per class
        grad_J_accum = np.zeros_like(learner.J)
        class_counts = np.zeros(n_classes)

        for idx in perm[:n_batch]:
            x = X_train[idx].copy()
            c = int(y_train[idx])

            # Generate trajectory with exchanges
            traj = thermalizer.trajectory(x, n_steps, n_exchanges_per_step)

            # Accumulate gradients from all exchanges
            sample_grad = np.zeros((n_dim, n_dim))
            n_total_exchanges = 0

            for step_idx in range(1, traj.n_steps):
                x_before = traj.steps[step_idx - 1].state
                exchanges = traj.steps[step_idx].exchanges

                for exchange in exchanges:
                    grad = learner.compute_exchange_gradient(x_before, exchange, c)
                    sample_grad += grad
                    n_total_exchanges += 1

            if n_total_exchanges > 0:
                sample_grad /= n_total_exchanges

            # Clip gradient
            grad_norm = np.linalg.norm(sample_grad)
            if grad_norm > 1.0:
                sample_grad = sample_grad / grad_norm

            grad_J_accum[c] += sample_grad
            class_counts[c] += 1

        # Update J for each class
        for c in range(n_classes):
            if class_counts[c] > 0:
                avg_grad = grad_J_accum[c] / class_counts[c]
                learner.J[c] -= lr * avg_grad
                # Enforce symmetry
                learner.J[c] = (learner.J[c] + learner.J[c].T) / 2

        # Compute metrics
        train_acc = learner.accuracy(X_train[:500], y_train[:500])
        test_acc = learner.accuracy(X_test, y_test) if X_test is not None else 0.0
        J_norm = np.linalg.norm(learner.J)

        history.epochs.append(epoch)
        history.train_acc.append(train_acc)
        history.test_acc.append(test_acc)
        history.J_norm.append(J_norm)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train={train_acc:5.1f}%, Test={test_acc:5.1f}%, ||J||={J_norm:.4f}")

    return history


def visualize_thermalizing_digit(X: np.ndarray, y: np.ndarray, thermalizer: ThermalPixelExchange,
                                  class_idx: int = 3, n_steps: int = 50, image_size: int = 14) -> None:
    """Show a digit thermalizing over time."""
    # Find a sample of the given class
    idx = np.where(y == class_idx)[0][0]
    x = X[idx].copy()

    # Generate trajectory
    traj = thermalizer.trajectory(x, n_steps, n_exchanges_per_step=100)

    # Verify mass conservation
    assert traj.total_mass_conserved, "Mass not conserved!"

    # Plot
    n_show = min(10, traj.n_steps)
    step_indices = np.linspace(0, traj.n_steps - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(2 * n_show, 2))
    for ax_idx, step_idx in enumerate(step_indices):
        img = traj.steps[step_idx].state.reshape(image_size, image_size)
        axes[ax_idx].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[ax_idx].set_title(f't={step_idx}')
        axes[ax_idx].axis('off')

    total_mass_start = traj.initial_state.sum()
    total_mass_end = traj.final_state.sum()
    plt.suptitle(f'Digit {class_idx} thermalizing (mass: {total_mass_start:.2f} → {total_mass_end:.2f})')
    plt.tight_layout()
    plt.savefig('paper_figures/thermalization_demo.png', dpi=150)
    print("Saved paper_figures/thermalization_demo.png")
    plt.close()


def visualize_results(learner: CouplingLearner, history: TrainingHistory, image_size: int = 14) -> None:
    """Visualize learned J matrices."""
    n_classes = learner.n_classes

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.epochs, history.train_acc, label='Train')
    axes[0].plot(history.epochs, history.test_acc, label='Test')
    axes[0].axhline(y=10, color='gray', linestyle=':', label='Random')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Classification Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.epochs, history.J_norm)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('||J||')
    axes[1].set_title('Total Coupling Norm')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('paper_figures/exchange_learning_curves.png', dpi=150)
    print("Saved paper_figures/exchange_learning_curves.png")
    plt.close()

    # Learned J matrices
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    vmax = np.abs(learner.J).max()
    for c in range(n_classes):
        ax = axes[c]
        im = ax.imshow(learner.J[c], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'J_{c}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Learned Coupling Matrices J_c (from exchange dynamics)', fontsize=12)
    plt.tight_layout()
    plt.savefig('paper_figures/exchange_learned_J.png', dpi=150)
    print("Saved paper_figures/exchange_learned_J.png")
    plt.close()

    # Spatial structure of couplings (center pixel)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    center_idx = (image_size // 2) * image_size + (image_size // 2)
    for c in range(n_classes):
        ax = axes[c]
        center_couplings = learner.J[c][center_idx, :].reshape(image_size, image_size)
        vmax_c = np.abs(center_couplings).max() or 1
        ax.imshow(center_couplings, cmap='RdBu_r', vmin=-vmax_c, vmax=vmax_c)
        ax.set_title(f'Class {c}')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot(image_size // 2, image_size // 2, 'ko', markersize=3)

    plt.suptitle('Center pixel couplings (learned from exchange dynamics)', fontsize=12)
    plt.tight_layout()
    plt.savefig('paper_figures/exchange_learned_J_spatial.png', dpi=150)
    print("Saved paper_figures/exchange_learned_J_spatial.png")
    plt.close()


def nearest_centroid_accuracy(X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               n_classes: int = 10) -> float:
    """Baseline: nearest centroid classifier."""
    centroids = np.zeros((n_classes, X_train.shape[1]))
    for c in range(n_classes):
        mask = (y_train == c)
        if mask.sum() > 0:
            centroids[c] = X_train[mask].mean(axis=0)

    correct = 0
    for i in range(len(X_test)):
        dists = np.linalg.norm(centroids - X_test[i], axis=1)
        if np.argmin(dists) == y_test[i]:
            correct += 1

    return correct / len(X_test) * 100


def center_within_class(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        n_classes: int = 10) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Center each class to have zero mean.

    This KILLS the centroid signal completely - after this, all class
    centroids are at zero, so nearest centroid MUST fail.

    The only discriminative signal left is SECOND-ORDER structure (correlations).
    """
    X_train_centered = X_train.copy()
    X_test_centered = X_test.copy()

    class_means = {}
    for c in range(n_classes):
        train_mask = (y_train == c)
        if train_mask.sum() > 0:
            class_mean = X_train[train_mask].mean(axis=0)
            class_means[c] = class_mean
            X_train_centered[train_mask] -= class_mean

        test_mask = (y_test == c)
        if test_mask.sum() > 0 and c in class_means:
            X_test_centered[test_mask] -= class_means[c]

    # Verify: per-class means should now be ~0
    for c in range(n_classes):
        train_mask = (y_train == c)
        if train_mask.sum() > 0:
            mean_norm = np.linalg.norm(X_train_centered[train_mask].mean(axis=0))
            assert mean_norm < 1e-5, f"Class {c} mean not zero! norm={mean_norm}"

    print("✓ All class means are now zero")
    return X_train_centered, X_test_centered, class_means


def main():
    print("=" * 70)
    print("COUPLING LEARNING WITH WITHIN-CLASS CENTERING")
    print("NO CHEATING: means removed, only second-order structure remains")
    print("=" * 70)

    image_size = 14
    n_dim = image_size * image_size
    n_classes = 10
    n_train = 5000
    n_test = 1000

    # Load data
    X_train_raw, y_train, X_test_raw, y_test = load_mnist_raw(n_train, n_test, image_size)

    # Baseline on RAW data
    print("\n--- Baseline: Nearest Centroid on RAW data ---")
    centroid_acc_raw = nearest_centroid_accuracy(X_train_raw, y_train, X_test_raw, y_test)
    print(f"Nearest Centroid (raw): {centroid_acc_raw:.1f}%")

    # CENTER WITHIN CLASS - this removes all first-order information!
    print("\n--- Within-Class Centering ---")
    print("Subtracting per-class means to KILL centroid signal...")
    X_train, X_test, class_means = center_within_class(
        X_train_raw, y_train, X_test_raw, y_test, n_classes
    )

    # Baseline on CENTERED data - should be ~random!
    print("\n--- Baseline: Nearest Centroid on CENTERED data ---")
    centroid_acc_centered = nearest_centroid_accuracy(X_train, y_train, X_test, y_test)
    print(f"Nearest Centroid (centered): {centroid_acc_centered:.1f}%")
    if centroid_acc_centered < 15:
        print("✓ Centroid signal KILLED - now at random chance!")
    else:
        print("⚠ WARNING: Centroid still working, centering may have failed")

    # Create thermalizer
    kT = 0.1
    thermalizer = ThermalPixelExchange(n_dim, kT=kT)

    # Demo: show thermalization on CENTERED data
    print("\n--- Thermalization Demo (centered data) ---")
    visualize_thermalizing_digit(X_train, y_train, thermalizer, class_idx=3,
                                  n_steps=100, image_size=image_size)

    # Create learner
    learner = CouplingLearner(n_dim, n_classes, kT=kT)

    # Train on CENTERED data
    print("\n--- Training via Exchange Observation (on centered data) ---")
    print("If this works, we're learning REAL second-order structure!")
    history = train_from_exchanges(
        learner, thermalizer,
        X_train, y_train,
        n_epochs=100,
        n_steps=5,
        n_exchanges_per_step=50,
        lr=0.01,
        samples_per_epoch=500,
        X_test=X_test,
        y_test=y_test
    )

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    final_test_acc = learner.accuracy(X_test, y_test)

    print(f"\nNearest Centroid (raw data):     {centroid_acc_raw:.1f}%")
    print(f"Nearest Centroid (centered):     {centroid_acc_centered:.1f}% (should be ~10%)")
    print(f"J Learning (centered):           {final_test_acc:.1f}%")
    print(f"Random baseline:                 10.0%")

    if final_test_acc > centroid_acc_centered + 5:
        print(f"\n✓ SUCCESS: J learning achieves {final_test_acc:.1f}% where centroids fail!")
        print("  This proves we're learning REAL second-order structure.")
    else:
        print(f"\n✗ J learning did not significantly beat random on centered data.")
        print("  Need to investigate gradient formulation.")

    # Visualize
    print("\n--- Visualizations ---")
    visualize_results(learner, history, image_size)

    np.save('learned_J_centered.npy', learner.J)
    print("Saved learned_J_centered.npy")

    # Run reconstruction demo (NO bias - pure J only!)
    print("\n--- Reconstruction Demo (J only, NO bias!) ---")
    reconstruct_from_J_only(learner.J, image_size, n_classes)

    return learner, history


def langevin_step_J_only(x: np.ndarray, J: np.ndarray,
                          J2: float, J4: float, kT: float, dt: float) -> np.ndarray:
    """
    One step of Langevin dynamics under the φ⁴+coupling potential.

    V(x) = J2 ||x||² + J4 ||x||⁴ + (1/2) x^T J x

    NO BIAS - only second-order coupling!

    dx = -∇V dt + √(2kT dt) η
    """
    # Gradient of potential (NO BIAS TERM!)
    grad_V = (2 * J2 * x +                    # φ² term (creates bistability)
              4 * J4 * (x ** 3) +             # φ⁴ term (stabilizes)
              J @ x)                           # coupling only!

    # Langevin update
    noise = np.random.randn(*x.shape) * np.sqrt(2 * kT * dt)
    return x - grad_V * dt + noise


def reconstruct_from_J_only(J: np.ndarray, image_size: int,
                             n_classes: int = 10, n_steps: int = 500,
                             J2: float = -1.0, J4: float = 0.5, kT: float = 0.3,
                             dt: float = 0.02):
    """
    Reconstruct from noise using ONLY learned couplings.

    NO CLASS MEANS / NO BIAS - this tests whether J alone
    creates meaningful structure.

    V(x) = J2 ||x||² + J4 ||x||⁴ + (1/2) x^T J_c x
    """
    n_dim = image_size * image_size

    fig, axes = plt.subplots(n_classes, 10, figsize=(15, 15))

    for c in range(n_classes):
        # Start from random noise
        x = np.random.randn(n_dim) * 0.5

        # Class-specific coupling ONLY - no bias!
        J_c = J[c]

        # Trajectory snapshots
        steps_to_show = [0, 5, 10, 20, 50, 100, 200, 300, 400, n_steps - 1]
        trajectory = [x.copy()]

        for t in range(1, n_steps):
            x = langevin_step_J_only(x, J_c, J2, J4, kT, dt)
            if t in steps_to_show:
                trajectory.append(x.copy())

        # Fill in missing steps if n_steps is smaller
        while len(trajectory) < 10:
            trajectory.append(trajectory[-1].copy())

        # Plot trajectory
        for i, frame in enumerate(trajectory[:10]):
            ax = axes[c, i]
            img = frame.reshape(image_size, image_size)
            ax.imshow(img, cmap='gray', vmin=-2, vmax=2)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f't={steps_to_show[i] if i < len(steps_to_show) else "end"}', fontsize=8)
            if i == 0:
                ax.set_ylabel(f'Class {c}', fontsize=10)

    plt.suptitle('Reconstruction from Noise using ONLY learned J\n(NO bias/templates - pure second-order structure)', fontsize=12)
    plt.tight_layout()
    plt.savefig('paper_figures/J_only_reconstruction.png', dpi=150)
    print("Saved paper_figures/J_only_reconstruction.png")
    plt.close()

    # Final reconstructions
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for c in range(n_classes):
        x = np.random.randn(n_dim) * 0.5
        J_c = J[c]

        for t in range(n_steps):
            x = langevin_step_J_only(x, J_c, J2, J4, kT, dt)

        ax = axes[c]
        img = x.reshape(image_size, image_size)
        ax.imshow(img, cmap='gray', vmin=-2, vmax=2)
        ax.set_title(f'Class {c}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Final States (t=500) - J only, NO bias', fontsize=12)
    plt.tight_layout()
    plt.savefig('paper_figures/J_only_final.png', dpi=150)
    print("Saved paper_figures/J_only_final.png")
    plt.close()


def reconstruct_full_model(J: np.ndarray, class_means: np.ndarray, image_size: int,
                            n_classes: int = 10, n_steps: int = 1000,
                            J2: float = -0.5, J4: float = 0.25, kT: float = 0.05,
                            dt: float = 0.05):
    """
    Generate digits using FULL model: J (correlations) + bias (templates).

    V_c(x) = J2||x||² + J4||x||⁴ + b_c·x + (1/2)x^T J_c x

    This is the complete thermodynamic potential with:
    - φ⁴ bistability
    - Class-specific bias (template)
    - Learned couplings (correlations)
    """
    n_dim = image_size * image_size

    fig, axes = plt.subplots(n_classes, 10, figsize=(15, 15))

    for c in range(n_classes):
        x = np.random.randn(n_dim) * 0.5
        J_c = J[c]
        b_c = -class_means[c]  # Bias attracts toward class template

        steps_to_show = [0, 5, 10, 20, 50, 100, 200, 300, 400, n_steps - 1]
        trajectory = [x.copy()]

        for t in range(1, n_steps):
            # Full potential gradient
            grad_V = (2 * J2 * x + 4 * J4 * (x ** 3) + b_c + J_c @ x)
            noise = np.random.randn(n_dim) * np.sqrt(2 * kT * dt)
            x = x - grad_V * dt + noise
            if t in steps_to_show:
                trajectory.append(x.copy())

        while len(trajectory) < 10:
            trajectory.append(trajectory[-1].copy())

        for i, frame in enumerate(trajectory[:10]):
            ax = axes[c, i]
            img = frame.reshape(image_size, image_size)
            ax.imshow(img, cmap='gray', vmin=-1.5, vmax=1.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f't={steps_to_show[i] if i < len(steps_to_show) else "end"}', fontsize=8)
            if i == 0:
                ax.set_ylabel(f'Class {c}', fontsize=10)

    plt.suptitle('Generation via Langevin Dynamics\nFull Model: φ⁴ + bias (template) + J (correlations)', fontsize=12)
    plt.tight_layout()
    plt.savefig('paper_figures/full_model_generation.png', dpi=150)
    print("Saved paper_figures/full_model_generation.png")
    plt.close()

    # Final results only
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    for c in range(n_classes):
        x = np.random.randn(n_dim) * 0.5
        J_c = J[c]
        b_c = -class_means[c]

        for t in range(n_steps):
            grad_V = (2 * J2 * x + 4 * J4 * (x ** 3) + b_c + J_c @ x)
            noise = np.random.randn(n_dim) * np.sqrt(2 * kT * dt)
            x = x - grad_V * dt + noise

        ax = axes[c]
        img = x.reshape(image_size, image_size)
        ax.imshow(img, cmap='gray', vmin=-1.5, vmax=1.5)
        ax.set_title(f'Class {c}')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('Generated Digits (Full Model: bias + learned J)', fontsize=12)
    plt.tight_layout()
    plt.savefig('paper_figures/generated_digits.png', dpi=150)
    print("Saved paper_figures/generated_digits.png")
    plt.close()


def run_reconstruction_only():
    """Load saved J matrices and run reconstruction demo."""
    print("Loading saved J matrices...")
    J = np.load('learned_J_centered.npy')

    # Infer dimensions from J shape
    n_classes, n_dim, _ = J.shape
    image_size = int(np.sqrt(n_dim))

    print(f"J shape: {J.shape} → {n_classes} classes, {image_size}x{image_size} images")

    # Compute class means from data
    X_train, y_train, _, _ = load_mnist_raw(5000, 1000, image_size)
    class_means = np.zeros((n_classes, n_dim))
    for c in range(n_classes):
        mask = (y_train == c)
        if mask.sum() > 0:
            class_means[c] = X_train[mask].mean(axis=0)

    print("\n1. J-only reconstruction (no bias)...")
    reconstruct_from_J_only(J, image_size, n_classes=n_classes)

    print("\n2. Full model generation (bias + J)...")
    reconstruct_full_model(J, class_means, image_size, n_classes=n_classes)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--reconstruct':
        run_reconstruction_only()
    else:
        main()
