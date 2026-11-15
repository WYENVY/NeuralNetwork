"""
train.py - train the from-scratch network on handwritten digits, with a live dashboard.

    python train.py              # 8x8 digits (built into scikit-learn, no download)
    python train.py --mnist      # 28x28 MNIST (downloads ~15 MB the first time)
    python train.py --no-live    # skip the window, just print progress
"""

# This training script loads a dataset, optionally augments it, trains a full
# neural network with gradient descent, and optionally renders a live dashboard.
import argparse
import numpy as np
from nn import NeuralNetwork, one_hot
from augment import augment_batch


# -----------------------------------------------------------------------------
# Dataset loading: support either the small 8x8 digits dataset or MNIST.
# -----------------------------------------------------------------------------
def load_data(use_mnist):
    """Load digit data and split it into train/test sets, returning flattened samples."""
    if use_mnist:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = X.astype(np.float64) / 255.0
        y = y.astype(int)
        side, n_train = 28, 60000
    else:
        from sklearn.datasets import load_digits
        d = load_digits()
        X, y = d.data / 16.0, d.target
        side, n_train = 8, 1400

    # Shuffle the data once to avoid ordering bias in the training split.
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    return X[:n_train], y[:n_train], X[n_train:], y[n_train:], side


# -----------------------------------------------------------------------------
# Main training loop.
# -----------------------------------------------------------------------------
def main():
    """Configure the run, train the model, and save weights to disk."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--refresh", type=int, default=None,
                    help="update the dashboard every N batches (default: once per epoch, "
                         "or every 300 batches on MNIST)")
    ap.add_argument("--out", default=None,
                    help="model filename (default: model_mnist.npz or model_digits.npz)")
    ap.add_argument("--augment", action="store_true",
                    help="randomly rotate/scale/shear/shift training digits (better on messy handwriting)")
    ap.add_argument("--no-live", action="store_true")
    args = ap.parse_args()

    # Load data and create the network architecture.
    Xtr, ytr, Xte, yte, side = load_data(args.mnist)
    Ytr = one_hot(ytr, 10)
    arch = [side * side, *args.hidden, 10]
    net = NeuralNetwork(arch, seed=0)
    print(f"train={len(Xtr)}  test={len(Xte)}  architecture={arch}")

    # Create a fixed "messy" copy of the test set to measure robustness under
    # small geometric distortions not seen in the clean evaluation set.
    Xmess = augment_batch(Xte, side, np.random.default_rng(123), max_rot=20, max_shift=0.12,
                          scale=(0.85, 1.15), max_shear=0.15)

    # Dashboard update cadence: once per epoch by default, but more frequently for MNIST.
    n_batches = int(np.ceil(len(Xtr) / args.batch))
    refresh = args.refresh or (300 if args.mnist else n_batches)

    # Optionally create the live dashboard. When disabled, the script just prints progress.
    dash = None
    if not args.no_live:
        import matplotlib.pyplot as plt
        plt.ion()
        from dashboard import Dashboard
        dash = Dashboard(net, Xte, yte, side, arch)

    # Training bookkeeping for running averages and dashboard updates.
    rng = np.random.default_rng(1)
    step, run_loss, run_n, last_loss = 0, 0.0, 0, float('nan')

    for epoch in range(1, args.epochs + 1):
        # Shuffle training examples each epoch to reduce ordering effects.
        order = rng.permutation(len(Xtr))

        for i in range(0, len(Xtr), args.batch):
            b = order[i:i + args.batch]
            xb = augment_batch(Xtr[b], side, rng) if args.augment else Xtr[b]

            # One stochastic gradient descent update on this mini-batch.
            run_loss += net.train_batch(xb, Ytr[b], args.lr)
            run_n += 1
            step += 1

            # Refresh the live dashboard periodically.
            if dash and step % refresh == 0:
                dash.update(step / n_batches, run_loss / run_n,
                            net.evaluate(Xtr[:2000], ytr[:2000]), net.evaluate(Xte, yte))
                last_loss, run_loss, run_n = run_loss / run_n, 0.0, 0

        if run_n:
            last_loss = run_loss / run_n

        # Report the training/test performance after each epoch.
        print(f"epoch {epoch:3d}  loss {last_loss:.4f}  "
              f"train acc {net.evaluate(Xtr[:2000], ytr[:2000]):.3f}  test acc {net.evaluate(Xte, yte):.3f}"
              f"  messy-test acc {net.evaluate(Xmess, yte):.3f}")
        run_loss, run_n = 0.0, 0

    # Save the trained weights to disk so they can be reused by draw.py.
    out = args.out or (("model_mnist" if args.mnist else "model_digits") + ("_aug.npz" if args.augment else ".npz"))
    np.savez(out, **{f"W{i}": l.W for i, l in enumerate(net.layers)},
             **{f"b{i}": l.b for i, l in enumerate(net.layers)}, side=side)
    print(f"saved {out}")

    # If live plotting was enabled, save the final dashboard image and keep the window open.
    if dash:
        dash.save("training_result.png")
        print("saved training_result.png")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
