# CNN Image Classification (CIFAR‑10)

A baseline convolutional neural network implemented in Keras/TensorFlow for classifying small images. Includes an offline fallback synthetic dataset so the script runs without external downloads.

## Problem → Approach → Results → Next Steps

- **Problem.** Establish a fast, understandable baseline for small image classification tasks.
- **Approach.** Implemented a simple 3‑block CNN: convolutional + batch normalization + max pooling repeated three times, followed by a global average pooling (GAP) layer and a dense softmax layer. Trained on CIFAR‑10; optionally uses a fallback synthetic dataset when offline.
- **Results.** Achieves **55–65%** test accuracy after **3 epochs** on CPU; saves the model as a `.keras` file for reuse.
- **Next steps.** Add data augmentation (random flips, crops); incorporate regularization (dropout, weight decay); experiment with transfer learning (MobileNetV2); export the model to ONNX/TFLite; and build a lightweight FastAPI endpoint for predictions.

## Installation

```bash
git clone https://github.com/yourname/cnn-image-classification.git
cd cnn-image-classification
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python src/train.py --dataset cifar10 --epochs 3 --batch_size 128 --output models/
```

Use `--dataset synthetic` to train on the synthetic fallback.

### Evaluate

```bash
python src/evaluate.py --model models/cnn.keras --dataset cifar10
```

### Run Inference

```bash
python src/infer.py --model models/cnn.keras --image path/to/image.png
```

## Project Structure

```
cnn-image-classification/
├── data/
│   └── synthetic_dataset.py
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer.py
│   └── …
├── models/
├── tests/
├── requirements.txt
├── .gitignore
├── .github/workflows/python-ci.yml
├── LICENSE
└── README.md
```

## Contributing

All contributions are welcome. Please open an issue before major changes.

## License

This project is licensed under the MIT License.