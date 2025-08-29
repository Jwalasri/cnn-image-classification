"""Unit tests for the CNN image classification project."""

from pathlib import Path

import numpy as np

from data.synthetic_dataset import get_synthetic_dataset
from src.model import create_model


def test_synthetic_dataset_shapes() -> None:
    X, y = get_synthetic_dataset(n_samples=50, num_classes=10, random_state=0)
    assert X.shape == (50, 32, 32, 3)
    assert y.shape == (50,)
    assert X.dtype == np.float32


def test_model_training() -> None:
    X, y = get_synthetic_dataset(n_samples=100, num_classes=10, random_state=1)
    X_flat = X.reshape((X.shape[0], -1))
    model = create_model(input_dim=X_flat.shape[1], num_classes=10)
    model.max_iter = 1
    model.fit(X_flat, y)
    preds = model.predict(X_flat)
    assert preds.shape == (100,)