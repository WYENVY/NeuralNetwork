"""
dashboard.py - a dark-theme live dashboard for watching the network learn.

Panels: stat header, loss + accuracy curves, a network diagram whose neurons
light up for a live sample, learned layer-1 weights, a confusion matrix,
and a strip of test predictions.
"""

# Matplotlib is used for the live dashboard visuals, and NumPy powers the array
# manipulations required to render activations, weights, and confusion data.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

# --- Theme colors ----------------------------------------------------------------
# Using a dark palette keeps the UI readable and makes the learning signals stand out.
BG, PANEL, GRID = "#0d1117", "#161b22", "#30363d"
TEXT, MUTED = "#e6edf3", "#8b949e"
BLUE, ORANGE, GREEN, RED, PURPLE = "#58a6ff", "#f78166", "#3fb950", "#f85149", "#bc8cff"

# Colormaps used by the network diagram and weight/confusion visualizations.
NODE_CMAP = LinearSegmentedColormap.from_list("node", ["#1c2430", "#1f6feb", "#58a6ff", "#f0f6fc"])
CONF_CMAP = LinearSegmentedColormap.from_list("conf", ["#161b22", "#1f3a5f", "#58a6ff", "#f0f6fc"])
W_CMAP = LinearSegmentedColormap.from_list("w", [ORANGE, "#161b22", BLUE])

# Global Matplotlib styling: dark backgrounds, muted labels, and consistent typography.
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "font.family": "DejaVu Sans", "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlecolor": TEXT, "axes.titlepad": 10,
})

# Maximum number of neurons displayed per hidden layer in the live network view.
MAX_NODES = 14


# -----------------------------------------------------------------------------
# Helper styling for dashboard subplots.
# -----------------------------------------------------------------------------
def _style(ax, title):
    """Apply the standard visual treatment to a dashboard panel."""
    ax.set_title(title, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, alpha=0.5, lw=0.6)


# -----------------------------------------------------------------------------
# Live network drawing: hidden activations and output probabilities.
# -----------------------------------------------------------------------------
def draw_network(ax, net, x, side, caption="input"):
    """Draw the live network diagram for one input vector x of shape (1, n_in).
    Returns the output probabilities for the current input.
    """
    # Clear the old drawing and set the panel background to match the dashboard theme.
    ax.clear()
    ax.axis("off")
    ax.set_facecolor(PANEL)
    ax.set_title("Live network activity", loc="left")

    # Run a forward pass so we can inspect the current activations and predictions.
    net.forward(x)
    acts = [l.act.forward(l.z)[0] for l in net.layers[:-1]]  # Hidden activations per layer
    probs = net.predict_proba(x)[0]

    # Spread the hidden-layer columns and output column across the graph width.
    n_hidden = len(acts)
    xs = np.linspace(0.2, 0.9, n_hidden + 1)  # hidden columns + output

    layers_xy, layers_idx = [], []
    for k, a in enumerate(acts):
        # Keep the display compact by tracking only the strongest neurons in each hidden layer.
        n = min(MAX_NODES, len(a))
        idx = np.argsort(-np.abs(a))[:n]
        idx = np.sort(idx)  # Sort so the output view is stable and readable.
        ys = np.linspace(0.9, 0.1, n) if n > 1 else np.array([0.5])
        layers_xy.append(np.c_[np.full(n, xs[k]), ys])
        layers_idx.append(idx)

    # Output layer uses a fixed set of 10 classes for digits / labels.
    out_ys = np.linspace(0.92, 0.08, 10)
    layers_xy.append(np.c_[np.full(10, xs[-1]), out_ys])
    layers_idx.append(np.arange(10))

    # Draw edges between neurons. Blue indicates positive weight influence,
    # orange indicates negative influence, and the line opacity reflects strength.
    for k in range(1, len(layers_xy)):
        W = net.layers[k].W
        src, dst = layers_idx[k - 1], layers_idx[k]
        sub = W[np.ix_(src, dst)]
        m = np.abs(sub).max() + 1e-9
        segs, cols = [], []
        for i, p in enumerate(layers_xy[k - 1]):
            for j, q in enumerate(layers_xy[k]):
                w = sub[i, j] / m
                if abs(w) < 0.25:
                    continue
                segs.append([p, q])
                cols.append((*matplotlib_rgb(BLUE if w > 0 else ORANGE), min(0.75, abs(w) * 0.7)))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.9, zorder=1))

    # Draw hidden-layer neurons, with brightness scaled by their activation strength.
    for k in range(n_hidden):
        a = acts[k][layers_idx[k]]
        norm = a / (a.max() + 1e-9) if a.max() > 0 else a
        for (px, py), v in zip(layers_xy[k], np.clip(norm, 0, 1)):
            ax.add_patch(Circle((px, py), 0.016, color=NODE_CMAP(v), ec=GRID, lw=0.8, zorder=3))
        ax.text(xs[k], 0.99, f"hidden {k + 1}", ha="center", color=MUTED, fontsize=9)
        ax.text(xs[k], 0.015, f"{len(acts[k])} neurons", ha="center", color=MUTED, fontsize=8)

    # Draw output neurons. The winning class gets a green outline.
    for (px, py), p, d in zip(layers_xy[-1], probs, range(10)):
        win = d == probs.argmax()
        ax.add_patch(Circle((px, py), 0.021, color=NODE_CMAP(p), ec=GREEN if win else GRID,
                            lw=2.2 if win else 0.8, zorder=3))
        ax.text(px + 0.04, py, f"{d}", va="center", fontsize=11,
                color=GREEN if win else MUTED, fontweight="bold" if win else "normal")
        ax.text(px + 0.075, py, f"{p * 100:4.1f}%", va="center", fontsize=8, color=MUTED)
    ax.text(xs[-1], 0.99, "output", ha="center", color=MUTED, fontsize=9)

    # Small inset showing the actual input image feeding the network.
    ins = ax.inset_axes([0.02, 0.36, 0.14, 0.28])
    ins.imshow(x.reshape(side, side), cmap="magma")
    ins.axis("off")
    ax.text(0.09, 0.32, caption, ha="center", va="top", color=MUTED, fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return probs


# -----------------------------------------------------------------------------
# Dashboard widget that updates over time while training runs.
# -----------------------------------------------------------------------------
class Dashboard:
    def __init__(self, net, Xte, yte, side, arch):
        """Create the full dashboard layout, charts, and live panel references."""
        self.net, self.Xte, self.yte, self.side = net, Xte, yte, side
        self.sample = 0
        self.hist = {"x": [], "loss": [], "tr": [], "te": []}

        # Build the main figure and reserve the subplot grid for each live metric panel.
        self.fig = plt.figure(figsize=(16, 9.2))
        try:
            self.fig.canvas.manager.set_window_title("Neural network from scratch")
        except Exception:
            pass

        gs = self.fig.add_gridspec(
            4, 3, height_ratios=[0.16, 1, 1, 0.55], width_ratios=[1, 1.45, 1],
            left=0.04, right=0.98, top=0.97, bottom=0.04, hspace=0.45, wspace=0.18
        )

        # --- Header panel ------------------------------------------------------
        self.ax_head = self.fig.add_subplot(gs[0, :])
        self.ax_head.axis("off")
        self.ax_head.text(0, 0.78, "Neural network from scratch", fontsize=20, fontweight="bold",
                          va="center", transform=self.ax_head.transAxes)
        self.ax_head.text(0, 0.22, "architecture  " + "  →  ".join(str(a) for a in arch),
                          fontsize=11, color=MUTED, va="center", transform=self.ax_head.transAxes)

        # Stats displayed in the header: epoch, loss, train accuracy, test accuracy.
        self.stats = {}
        for i, (key, label, color) in enumerate(
                [("epoch", "EPOCH", TEXT), ("loss", "LOSS", RED),
                 ("tr", "TRAIN ACC", BLUE), ("te", "TEST ACC", GREEN)]):
            x = 0.52 + i * 0.12
            self.ax_head.text(x, 0.92, label, fontsize=9, color=MUTED, va="center",
                              transform=self.ax_head.transAxes)
            self.stats[key] = self.ax_head.text(x, 0.4, "-", fontsize=22, fontweight="bold",
                                                color=color, va="center",
                                                transform=self.ax_head.transAxes)

        # --- Main dashboard panels --------------------------------------------
        self.ax_loss = self.fig.add_subplot(gs[1, 0])
        _style(self.ax_loss, "Loss")

        self.ax_acc = self.fig.add_subplot(gs[2, 0])
        _style(self.ax_acc, "Accuracy")
        self.ax_acc.set_xlabel("epoch")
        self.ax_acc.set_ylim(0.0, 1.02)

        self.ax_net = self.fig.add_subplot(gs[1:3, 1])
        self.ax_net.axis("off")

        self.ax_w = self.fig.add_subplot(gs[1, 2])
        self.ax_cm = self.fig.add_subplot(gs[2, 2])

        (self.l_loss,) = self.ax_loss.plot([], [], color=RED, lw=2.2)
        self.f_loss = None
        (self.l_tr,) = self.ax_acc.plot([], [], color=BLUE, lw=2, label="train")
        (self.l_te,) = self.ax_acc.plot([], [], color=GREEN, lw=2.2, label="test")
        self.ax_acc.legend(loc="lower right", frameon=False, labelcolor=TEXT)

        # --- Prediction strip at the bottom ------------------------------------
        self.n_pred = 12
        sub = gs[3, :].subgridspec(1, self.n_pred, wspace=0.12)
        self.pred_axes = [self.fig.add_subplot(sub[0, i]) for i in range(self.n_pred)]

    # -------------------------------------------------------------------------
    # Panel renderers.
    # -------------------------------------------------------------------------
    def _draw_network(self):
        """Render the live activation map and current sample input."""
        i = self.sample % len(self.Xte)
        draw_network(self.ax_net, self.net, self.Xte[i].reshape(1, -1), self.side,
                     f"input\ntrue: {self.yte[i]}")

    def _draw_weights(self):
        """Display the first-layer weight patterns as learned feature detectors."""
        ax = self.ax_w
        ax.clear()
        ax.axis("off")
        ax.set_title("What layer-1 neurons look for", loc="left")

        W = self.net.layers[0].W
        s, g = self.side, 1
        n_rows = n_cols = 4
        canvas = np.full((n_rows * (s + g) - g, n_cols * (s + g) - g), np.nan)

        for k in range(n_rows * n_cols):
            r, c = divmod(k, n_cols)
            canvas[r * (s + g): r * (s + g) + s, c * (s + g): c * (s + g) + s] = W[:, k].reshape(s, s)

        lim = np.abs(W[:, :16]).max()
        cm = W_CMAP.copy()
        cm.set_bad(PANEL)
        ax.imshow(canvas, cmap=cm, vmin=-lim, vmax=lim)

    def _draw_confusion(self):
        """Compute and visualize the confusion matrix for the test set."""
        ax = self.ax_cm
        ax.clear()
        ax.set_title("Confusion matrix (test set)", loc="left")

        pred = self.net.predict(self.Xte)
        cm = np.zeros((10, 10), dtype=int)
        np.add.at(cm, (self.yte, pred), 1)

        ax.imshow(cm, cmap=CONF_CMAP, aspect="equal")
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color=BG if cm[i, j] > cm.max() * 0.5 else (RED if i != j else MUTED))

        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.tick_params(labelsize=8, length=0)
        ax.set_xlabel("predicted", fontsize=9)
        ax.set_ylabel("true", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)

    def _draw_predictions(self):
        """Show a strip of test-sample predictions and whether each is correct."""
        probs = self.net.predict_proba(self.Xte[:self.n_pred])
        for k, ax in enumerate(self.pred_axes):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

            p = probs[k].argmax()
            ok = p == self.yte[k]
            ax.imshow(self.Xte[k].reshape(self.side, self.side), cmap="magma")

            for s in ax.spines.values():
                s.set_edgecolor(GREEN if ok else RED)
                s.set_linewidth(2.5)

            ax.set_title(f"{p}  ·  {probs[k][p] * 100:.0f}%", fontsize=10,
                         color=GREEN if ok else RED, fontweight="bold", pad=4)

    # -------------------------------------------------------------------------
    # Update loop used during training.
    # -------------------------------------------------------------------------
    def update(self, epoch, loss, train_acc, test_acc):
        """Refresh all plots with the most recent training metrics."""
        h = self.hist
        h["x"].append(epoch)
        h["loss"].append(loss)
        h["tr"].append(train_acc)
        h["te"].append(test_acc)

        # Update the loss trace and shaded area under the curve.
        self.l_loss.set_data(h["x"], h["loss"])
        if self.f_loss is not None:
            self.f_loss.remove()
        self.f_loss = self.ax_loss.fill_between(h["x"], h["loss"], color=RED, alpha=0.12)
        self.ax_loss.set_xlim(0, max(1, epoch))
        self.ax_loss.set_ylim(0, max(h["loss"]) * 1.08)

        # Update the train/test accuracy curves.
        self.l_tr.set_data(h["x"], h["tr"])
        self.l_te.set_data(h["x"], h["te"])
        lo = max(0.0, min(min(h["tr"]), min(h["te"])) - 0.03)
        self.ax_acc.set_xlim(0, max(1, epoch))
        self.ax_acc.set_ylim(lo, 1.01)

        # Refresh the numeric stat labels in the header.
        self.stats["epoch"].set_text(f"{epoch:.1f}" if epoch % 1 else f"{int(epoch)}")
        self.stats["loss"].set_text(f"{loss:.4f}")
        self.stats["tr"].set_text(f"{train_acc * 100:.1f}%")
        self.stats["te"].set_text(f"{test_acc * 100:.1f}%")

        # Re-render all diagnostic panels for the current training snapshot.
        self._draw_network()
        self._draw_weights()
        self._draw_confusion()
        self._draw_predictions()

        self.sample += 1
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def save(self, path):
        """Persist the current dashboard view to a file."""
        self.fig.savefig(path, dpi=130, facecolor=BG)


# -----------------------------------------------------------------------------
# Convert a hex color into normalized RGB floats suitable for Matplotlib colors.
# -----------------------------------------------------------------------------
def matplotlib_rgb(hex_color):
    """Map a 6-digit hex color like '#58a6ff' to (r, g, b) in [0, 1]."""
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
