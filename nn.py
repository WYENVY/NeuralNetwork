"""
nn.py - a neural network from scratch.

Only NumPy is used, and only for array storage and matrix arithmetic.
Every piece of the learning machinery is written by hand:
neurons (Dense layers), weights, biases, activations, forward propagation,
loss, backpropagation, and gradient-descent updates.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Activation functions: each implements forward(z) and backward(z, grad_out).
# These provide the non-linearities used between dense layers.
# ---------------------------------------------------------------------------
class ReLU:
    """Rectified Linear Unit: keeps positive values and zeroes out negatives."""
    def forward(self, z):
        return np.maximum(0.0, z)

    def backward(self, z, grad_out):
        return grad_out * (z > 0)


class Sigmoid:
    """Smooth squashing function in the range (0, 1)."""
    def forward(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def backward(self, z, grad_out):
        s = self.forward(z)
        return grad_out * s * (1.0 - s)


class Tanh:
    """Zero-centered activation that often helps hidden layers learn faster."""
    def forward(self, z):
        return np.tanh(z)

    def backward(self, z, grad_out):
        return grad_out * (1.0 - np.tanh(z) ** 2)


class Identity:
    """Used on the output layer because the loss function handles softmax itself."""
    def forward(self, z):
        return z

    def backward(self, z, grad_out):
        return grad_out


# Lookup table for activation names to their implementations.
ACTIVATIONS = {"relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh, "identity": Identity}


# ---------------------------------------------------------------------------
# Softmax + cross-entropy loss, implemented in a numerically stable fused form.
# This gives the clean gradient: probabilities minus one-hot labels.
# ---------------------------------------------------------------------------
def softmax(logits):
    """Compute softmax probabilities with a stable shift to avoid overflow."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    """Cross-entropy loss paired with a softmax output layer."""
    def forward(self, logits, y_onehot):
        self.probs = softmax(logits)
        self.y = y_onehot
        n = logits.shape[0]
        return -np.sum(y_onehot * np.log(self.probs + 1e-12)) / n

    def backward(self):
        """Gradient w.r.t. logits is (probs - y) / batch_size."""
        return (self.probs - self.y) / self.y.shape[0]


# ---------------------------------------------------------------------------
# A single dense layer handling the affine transform and activation.
# ---------------------------------------------------------------------------
class Dense:
    """
    A fully connected layer of `n_out` neurons.
        z = x @ W + b          (weighted sum + bias)
        a = activation(z)      (non-linearity)
    """
    def __init__(self, n_in, n_out, activation="relu", rng=None):
        rng = rng or np.random.default_rng()

        # He init for ReLU helps maintain stable signal variance; Xavier works better
        # for non-ReLU activations. This keeps training from exploding or collapsing.
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = rng.standard_normal((n_in, n_out)) * scale
        self.b = np.zeros((1, n_out))
        self.act = ACTIVATIONS[activation]()

        # Momentum buffers for SGD with momentum.
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        """Compute the layer output and cache inputs needed for backpropagation."""
        self.x = x
        self.z = x @ self.W + self.b
        return self.act.forward(self.z)

    def backward(self, grad_a):
        """
        grad_a: derivative of the loss w.r.t. this layer's activation output.
        Returns derivative w.r.t. the previous layer's activation.

        Chain rule:
            dz = grad_a * act'(z)
            dW = x^T @ dz
            db = sum(dz over batch)
            dx = dz @ W^T
        """
        dz = self.act.backward(self.z, grad_a)
        self.dW = self.x.T @ dz
        self.db = dz.sum(axis=0, keepdims=True)
        return dz @ self.W.T

    def step(self, lr, momentum=0.9):
        """Apply momentum-based gradient descent update to weights and biases."""
        self.vW = momentum * self.vW - lr * self.dW
        self.vb = momentum * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


# ---------------------------------------------------------------------------
# Full feed-forward network assembled from Dense layers.
# ---------------------------------------------------------------------------
class NeuralNetwork:
    """A simple multilayer perceptron that supports forward, backprop, and training."""
    def __init__(self, sizes, hidden_activation="relu", seed=0):
        """sizes e.g. [64, 32, 16, 10]: input, hidden..., output."""
        rng = np.random.default_rng(seed)
        self.layers = []
        for i in range(len(sizes) - 1):
            last = i == len(sizes) - 2
            act = "identity" if last else hidden_activation
            self.layers.append(Dense(sizes[i], sizes[i + 1], act, rng))
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        """Propagate the input through all layers and return the final logits."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict_proba(self, x):
        """Return class probabilities for the given input batch."""
        return softmax(self.forward(x))

    def predict(self, x):
        """Return the predicted class index for each row in x."""
        return self.forward(x).argmax(axis=1)

    def backward(self):
        """Backpropagate the loss gradient through the network."""
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_batch(self, x, y_onehot, lr, momentum=0.9):
        """Run one forward pass, one backward pass, and one optimizer step."""
        logits = self.forward(x)
        loss = self.loss_fn.forward(logits, y_onehot)
        self.backward()
        for layer in self.layers:
            layer.step(lr, momentum)
        return loss

    def evaluate(self, x, y_labels):
        """Accuracy on a labeled dataset."""
        return float((self.predict(x) == y_labels).mean())


def one_hot(labels, n_classes):
    """Convert integer labels into one-hot vectors."""
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


# ---------------------------------------------------------------------------
# Gradient checking: verify backprop against finite differences.
# This is a numerical sanity check, not the actual training mechanism.
# ---------------------------------------------------------------------------
def gradient_check(eps=1e-5):
    """Compare analytic backprop to numerical derivatives; print PASS/FAIL."""
    rng = np.random.default_rng(1)
    net = NeuralNetwork([5, 4, 3], hidden_activation="tanh", seed=2)
    x = rng.standard_normal((6, 5))
    y = one_hot(rng.integers(0, 3, 6), 3)

    net.loss_fn.forward(net.forward(x), y)
    net.backward()

    worst = 0.0
    for layer in net.layers:
        for param, grad in ((layer.W, layer.dW), (layer.b, layer.db)):
            it = np.nditer(param, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                orig = param[idx]

                # Perturb the parameter up and down and compare finite-difference gradient.
                param[idx] = orig + eps
                lp = net.loss_fn.forward(net.forward(x), y)
                param[idx] = orig - eps
                lm = net.loss_fn.forward(net.forward(x), y)
                param[idx] = orig

                num = (lp - lm) / (2 * eps)
                denom = max(1e-8, abs(num) + abs(grad[idx]))
                worst = max(worst, abs(num - grad[idx]) / denom)

    print(f"gradient check: worst relative error = {worst:.2e}",
          "PASS" if worst < 1e-6 else "FAIL")
    return worst < 1e-6


if __name__ == "__main__":
    gradient_check()
