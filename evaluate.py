"""Evaluation script for the image classification project.

Loads a trained model file and evaluates it on a specified dataset.
Prints the classification accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from data.synthetic_dataset import load_dataset, NUM_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained image classifier.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model file (.joblib).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "cifar10"],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=2000,
        help="Number of samples to use when loading the dataset.",
    )
    args = parser.parse_args()
    # Load model
    model = joblib.load(args.model)
    # Load dataset
    if args.dataset == "synthetic":
        X, y = load_dataset("synthetic", n_samples=args.n_samples, random_state=1)
    else:
        X, y = load_dataset("cifar10")
        n = min(len(X), args.n_samples)
        X = X[:n]
        y = y[:n]
    X_flat = X.reshape((X.shape[0], -1))
    preds = model.predict(X_flat)
    accuracy = float((preds == y).mean())
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()