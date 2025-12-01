"""
Shared utilities for the Criticality Engine.

This module provides common data loading and preprocessing functions
used across trajectory estimation, denoising, and comparison scripts.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Cache for MNIST data to avoid repeated downloads
_MNIST_CACHE: dict[str, tuple[NDArray, NDArray]] | None = None


def _fetch_mnist() -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """
    Fetch MNIST dataset (cached).

    Returns:
        (X, y) tuple where X is (70000, 784) float32 and y is (70000,) int64
    """
    global _MNIST_CACHE

    if _MNIST_CACHE is None:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
        _MNIST_CACHE = (X, y)

    return _MNIST_CACHE


def downsample_images(
    X: NDArray[np.float32],
    image_size: int = 14
) -> NDArray[np.float32]:
    """
    Downsample 28x28 images to smaller size using average pooling.

    Args:
        X: Images of shape (n, 784) in [0, 1]
        image_size: Target size (default 14x14)

    Returns:
        Downsampled images of shape (n, image_size*image_size) in [-1, 1]
    """
    n = len(X)
    X_full = X.reshape(-1, 28, 28)
    factor = 28 // image_size
    X_down = np.zeros((n, image_size, image_size), dtype=np.float32)

    for i in range(image_size):
        for j in range(image_size):
            X_down[:, i, j] = X_full[:,
                i*factor:(i+1)*factor,
                j*factor:(j+1)*factor
            ].mean(axis=(1, 2))

    # Scale to [-1, 1]
    return 2 * X_down.reshape(n, -1) - 1


def load_mnist(
    n_train: int = 5000,
    n_test: int = 1000,
    image_size: int = 14
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.float32], NDArray[np.int64]]:
    """
    Load MNIST with train/test split and preprocessing.

    Uses first n_train samples for training and last n_test samples
    from the dataset for testing (guaranteed no overlap).

    Args:
        n_train: Number of training samples (from start of dataset)
        n_test: Number of test samples (from end of dataset)
        image_size: Size to downsample images to (default 14x14)

    Returns:
        (X_train, y_train, X_test, y_test) tuple
        - X arrays are float32 in [-1, 1], shape (n, image_size*image_size)
        - y arrays are int64 in [0, 9]
    """
    X, y = _fetch_mnist()

    # First n_train for training, last n_test for testing (no overlap)
    train_indices = np.arange(n_train)
    test_indices = np.arange(len(X) - n_test, len(X))

    X_train = downsample_images(X[train_indices], image_size)
    y_train = y[train_indices]
    X_test = downsample_images(X[test_indices], image_size)
    y_test = y[test_indices]

    print(f"Train: {len(X_train)}, Test: {len(X_test)} (holdout from end of dataset)")

    return X_train, y_train, X_test, y_test


def load_digit_samples(
    digit: int,
    n_samples: int = 5,
    image_size: int = 14
) -> NDArray[np.float32]:
    """
    Load samples of a specific digit class from MNIST.

    Args:
        digit: Digit class (0-9)
        n_samples: Number of samples to load
        image_size: Size to downsample to

    Returns:
        Array of shape (n_samples, image_size*image_size) in [-1, 1]
    """
    X, y = _fetch_mnist()

    mask = (y == digit)
    digit_images = X[mask][:n_samples]

    return downsample_images(digit_images, image_size)
