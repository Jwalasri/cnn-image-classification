"""Synthetic dataset utilities for image classification.

This module provides helper functions to generate and load simple
image datasets suitable for classification experiments.  Since deep
learning frameworks are not available in this environment, the
synthetic dataset is used as a fallback for training and testing.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


IMAGE_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


def get_synthetic_dataset(
    n_samples: int = 1000, num_classes: int = NUM_CLASSES, random_state: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic image dataset.

    The images are random noise and the labels are uniformly drawn
    integers.  While not meaningful for real classification, this
    synthetic dataset is sufficient to exercise the training and
    inference pipeline.

    Parameters
    ----------
    n_samples:
        Number of samples to generate.
    num_classes:
        Number of distinct classes (labels range from 0 to num_classes‑1).
    random_state:
        Optional seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A pair (X, y) where X has shape (n_samples, 32, 32, 3) and
        y has shape (n_samples,).
    """
    rng = np.random.default_rng(random_state)
    X = rng.random((n_samples, *IMAGE_SHAPE), dtype=np.float32)
    y = rng.integers(0, num_classes, size=n_samples, dtype=np.int64)
    return X, y


def load_dataset(name: str, n_samples: int = 1000, random_state: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset by name.

    Currently supports:

    - ``cifar10``: attempts to load from keras.  If unavailable, falls back to synthetic.
    - ``synthetic``: generates a random dataset.

    Parameters
    ----------
    name:
        Dataset name ('cifar10' or 'synthetic').
    n_samples:
        Number of samples to load (only applicable for synthetic dataset).  For
        ``cifar10`` the full 50k training set is loaded if available.
    random_state:
        Random seed for synthetic dataset.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A pair (X, y) containing the images and labels.
    """
    name = name.lower()
    if name == "cifar10":
        try:
            # Attempt to import keras and load the CIFAR‑10 dataset
            from tensorflow.keras.datasets import cifar10  # type: ignore

            (X_train, y_train), _ = cifar10.load_data()
            # Flatten labels
            y_train = y_train.flatten()
            return X_train.astype(np.float32) / 255.0, y_train.astype(np.int64)
        except Exception:
            # Fall back to synthetic dataset
            return get_synthetic_dataset(n_samples=n_samples, num_classes=NUM_CLASSES, random_state=random_state)
    elif name == "synthetic":
        return get_synthetic_dataset(n_samples=n_samples, num_classes=NUM_CLASSES, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset: {name}")