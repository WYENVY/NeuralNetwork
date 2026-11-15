# Neural Network Digit Classifier

A small, from-scratch neural network project for classifying handwritten digits using NumPy. It includes:

- a custom dense neural network with backpropagation
- live training metrics in a dark-mode dashboard
- optional data augmentation for better robustness
- an interactive drawing panel to test the trained model

## Features

- Train on the 8x8 sklearn digits dataset or on MNIST
- Custom neural network implementation in pure NumPy
- Softmax + cross-entropy loss
- ReLU hidden layers and momentum-based optimization
- Optional affine augmentation (rotation, scale, shear, translation)
- Real-time dashboard for loss, accuracy, confusion matrix, and live activations
- Interactive digit drawing app to test predictions on handwritten input

## Project structure

- `nn.py` — neural network implementation, activations, loss, and gradient check
- `train.py` — training entry point and dataset loading
- `augment.py` — data augmentation utilities for image jittering
- `dashboard.py` — visualisation dashboard for live training progress
- `draw.py` — interactive drawing and prediction interface

## Requirements

Install the Python dependencies before training or drawing:

```bash
pip install numpy matplotlib scipy scikit-learn
```

If you want a clean environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install numpy matplotlib scipy scikit-learn
```

## Quick start

### Train on the built-in digits dataset

```bash
python train.py
```

This uses the 8x8 sklearn digits dataset and saves a model file such as:

- `model_digits.npz`
- or `model_digits_aug.npz` if augmentation is enabled

### Train on MNIST

```bash
python train.py --mnist
```

This downloads the MNIST dataset the first time it is used.

### Train with augmentation

```bash
python train.py --augment
```

This applies random rotations, shifts, scaling, and shearing to training examples to help the model generalise better to imperfect handwriting.

### Run without the live dashboard

```bash
python train.py --no-live
```

This still trains the model but skips the animated dashboard and only prints progress to the terminal.

### Test the model by drawing digits

```bash
python draw.py
```

This will load the best available saved model from the working directory and allow you to draw a digit in the window. The model predicts the digit and shows confidence.

You can also specify a particular model file:

```bash
python draw.py --model model_mnist.npz
```

## Model output

Training writes a `.npz` archive containing the learned weights and biases as well as the `side` parameter used by the model.

Example saved files:

- `model_digits.npz`
- `model_digits_aug.npz`
- `model_mnist.npz`
- `model_mnist_aug.npz`

## Dashboard behavior

When live mode is enabled, the dashboard shows:

- epoch and loss
- train/test accuracy
- live network activity
- first-layer learned weights
- confusion matrix on the test set
- prediction strip for sample digits

## Useful commands

```bash
python train.py --mnist --epochs 20 --lr 0.05 --batch 64
python train.py --augment --no-live
python draw.py --model model_digits.npz
```

## Troubleshooting

- If you see a missing dependency error, install the requirements listed above.
- If no model is found, run `python train.py` first to generate one.
- If the dashboard appears blank or does not update, make sure matplotlib is installed correctly and that a graphical backend is available.

## License

This project is licensed under the MIT License. 


