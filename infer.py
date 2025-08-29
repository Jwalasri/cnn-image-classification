"""Run inference with a trained image classifier.

This script loads a joblib model, reads an image file, resizes it to
32×32 pixels and predicts its class.  It prints the predicted label
and probability score (if available).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_image(path: Path | str) -> np.ndarray:
    """Load an image file and convert it to a 32×32×3 NumPy array."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((32, 32))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a single image using a trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.joblib).")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file to classify.")
    args = parser.parse_args()
    model = joblib.load(args.model)
    img = load_image(args.image)
    img_flat = img.reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(img_flat)[0]
        pred_idx = int(np.argmax(probs))
        pred_prob = float(probs[pred_idx])
    else:
        pred_idx = int(model.predict(img_flat)[0])
        # With no probability estimates, set probability to 1.0 for predicted class
        pred_prob = 1.0
    label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)
    print(f"Predicted: {label} (prob={pred_prob:.2f})")


if __name__ == "__main__":
    main()