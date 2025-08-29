"""Model definition for the image classification project.

Because this environment does not include deep learning frameworks
such as TensorFlow or PyTorch, we implement a simple multilayer
perceptron (MLP) classifier using scikit‑learn.  The MLP operates on
flattened 32×32×3 images and outputs probabilities over 10 classes.
"""

from __future__ import annotations

from typing import Tuple

from sklearn.neural_network import MLPClassifier


def create_model(input_dim: int = 32 * 32 * 3, num_classes: int = 10) -> MLPClassifier:
    """Construct an MLP classifier for image data.

    Parameters
    ----------
    input_dim:
        Dimensionality of the flattened input (default 3072 for 32×32×3 images).
    num_classes:
        Number of output classes (default 10).

    Returns
    -------
    MLPClassifier
        A scikit‑learn multilayer perceptron classifier.
    """
    # Two hidden layers with ReLU activation; training for a small number of
    # iterations yields a reasonable baseline on the synthetic dataset.
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=10,
    )
    return model