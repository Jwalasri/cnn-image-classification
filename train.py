"""Training script for the image classification project.

This script trains a simple multilayer perceptron (MLP) on either
the CIFAR‑10 dataset or a synthetic fallback.  The trained model is
saved to the specified output directory as a joblib file.  Although
the README mentions a convolutional neural network, we rely on an MLP
due to the absence of deep learning libraries in this environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from data.synthetic_dataset import load_dataset, IMAGE_SHAPE, NUM_CLASSES
from .model import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP classifier for small images.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "cifar10"],
        help="Dataset to train on. 'cifar10' will fall back to synthetic if unavailable.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (maps to max_iter for the MLP).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (unused for MLP, kept for API compatibility).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "models"),
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of samples to generate for synthetic dataset.",
    )
    args = parser.parse_args()
    # Load dataset
    if args.dataset == "synthetic":
        X, y = load_dataset("synthetic", n_samples=args.n_samples, random_state=0)
    else:
        X, y = load_dataset("cifar10")
        # Limit number of samples to speed up training
        n = min(len(X), args.n_samples)
        X = X[:n]
        y = y[:n]
    # Flatten images for MLP input
    X_flat = X.reshape((X.shape[0], -1))
    # Create model
    model = create_model(input_dim=X_flat.shape[1], num_classes=NUM_CLASSES)
    # Map epochs to max_iter
    model.max_iter = args.epochs
    # Fit model
    model.fit(X_flat, y)
    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cnn.joblib"
    joblib.dump(model, model_path)
    print(f"Trained model saved to {model_path}")


if __name__ == "__main__":
    main()