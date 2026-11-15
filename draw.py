"""
draw.py - draw a digit and watch the network think, using the same dashboard style as training.

    python draw.py    # uses the best model it finds (MNIST + augmented first)
    python draw.py --model model_mnist.npz

Left-drag = draw      Right-click or 'c' = clear
"""

# This script loads a saved neural network, lets the user sketch a single digit,
# and then shows the model's prediction and confidence in real time.
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from nn import NeuralNetwork
from dashboard import (BG, PANEL, TEXT, MUTED, GREEN, BLUE, GRID, _style, draw_network)

# -----------------------------------------------------------------------------
# Command-line setup and model loading.
# -----------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None)
args = ap.parse_args()

# Pick the best available saved model, preferring MNIST-augmented versions when present.
path = args.model or next((p for p in ("model_mnist_aug.npz", "model_mnist.npz", "model_digits_aug.npz",
                                       "model_digits.npz", "model.npz")
                           if os.path.exists(p)), None)
if not path or not os.path.exists(path):
    raise SystemExit("No model found. Run train.py first (it saves model_mnist.npz / model_digits.npz).")

# Load saved weights and reconstruct the same network architecture.
data = np.load(path)
side = int(data["side"])
n_layers = len([k for k in data.files if k.startswith("W")])
sizes = [data["W0"].shape[0]] + [data[f"W{i}"].shape[1] for i in range(n_layers)]
net = NeuralNetwork(sizes)
for i, layer in enumerate(net.layers):
    layer.W, layer.b = data[f"W{i}"], data[f"b{i}"]
print(f"loaded {path}  architecture={sizes}")

# -----------------------------------------------------------------------------
# Drawing canvas: a large blank area where the user draws a digit.
# -----------------------------------------------------------------------------
RES = 224              # drawing canvas resolution
BRUSH = 9.0            # brush radius in pixels on the canvas
canvas = np.zeros((RES, RES))
yy, xx = np.mgrid[0:RES, 0:RES]
state = {"drawing": False, "last": 0.0, "last_xy": None}


# -----------------------------------------------------------------------------
# Preprocessing pipeline: convert a user sketch into the same format as training data.
# -----------------------------------------------------------------------------
def preprocess(c):
    """Turn the drawing into what the network was trained on: crop, normalize, center, and resize."""
    if c.max() < 0.05:
        return np.zeros((side, side))

    # Crop the stroke tightly to reduce empty canvas space before scaling.
    rows, cols = np.where(c > 0.1)
    c = c[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = c.shape
    n = max(h, w)

    # Place the cropped drawing in the center of a square canvas.
    sq = np.zeros((n, n))
    sq[(n - h) // 2:(n - h) // 2 + h, (n - w) // 2:(n - w) // 2 + w] = c

    # Resize the square to the standard MNIST-style box size inside the network input.
    box = 20 if side >= 28 else side - 1
    sq = ndimage.zoom(sq, box / n, order=1)

    # Place the resized image into the input grid and center-of-mass shift it.
    out = np.zeros((side, side))
    o = (side - sq.shape[0]) // 2
    out[o:o + sq.shape[0], o:o + sq.shape[1]] = sq[:side, :side]
    cy, cx = ndimage.center_of_mass(out)
    out = ndimage.shift(out, (side / 2 - cy - 0.5, side / 2 - cx - 0.5), order=1)

    return np.clip(out / max(out.max(), 1e-6), 0, 1)


# -----------------------------------------------------------------------------
# Figure layout: dashboard-style UI with drawing panel, live network, and predictions.
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 8.6))
try:
    fig.canvas.manager.set_window_title("Draw a digit")
except Exception:
    pass

gs = fig.add_gridspec(3, 3, height_ratios=[0.16, 1, 1], width_ratios=[1, 1.45, 1],
                      left=0.04, right=0.98, top=0.97, bottom=0.05, hspace=0.42, wspace=0.18)

# Header with prediction and confidence text.
ax_head = fig.add_subplot(gs[0, :])
ax_head.axis("off")
ax_head.text(0, 0.78, "Draw a digit", fontsize=20, fontweight="bold", va="center",
             transform=ax_head.transAxes)
ax_head.text(0, 0.22, "left-drag to draw   ·   right-click or C to clear   ·   draw big and centred",
             fontsize=11, color=MUTED, va="center", transform=ax_head.transAxes)
ax_head.text(0.52, 0.92, "PREDICTION", fontsize=9, color=MUTED, va="center", transform=ax_head.transAxes)
t_pred = ax_head.text(0.52, 0.4, "-", fontsize=26, fontweight="bold", color=GREEN, va="center",
                      transform=ax_head.transAxes)
ax_head.text(0.64, 0.92, "CONFIDENCE", fontsize=9, color=MUTED, va="center", transform=ax_head.transAxes)
t_conf = ax_head.text(0.64, 0.4, "-", fontsize=26, fontweight="bold", color=BLUE, va="center",
                      transform=ax_head.transAxes)

# Left panel: user's drawing canvas.
ax_draw = fig.add_subplot(gs[1:3, 0])
ax_draw.set_title("Your drawing", loc="left")
img = ax_draw.imshow(canvas, cmap="magma", vmin=0, vmax=1)
ax_draw.set_xticks([])
ax_draw.set_yticks([])
for s in ax_draw.spines.values():
    s.set_edgecolor(GRID)
    s.set_linewidth(1.5)

# Center panel: network activity diagram.
ax_net = fig.add_subplot(gs[1:3, 1])
ax_net.axis("off")

# Right upper panel: what the network sees after preprocessing.
ax_see = fig.add_subplot(gs[1, 2])
ax_see.set_title(f"What the network sees ({side}×{side})", loc="left")
see = ax_see.imshow(np.zeros((side, side)), cmap="magma", vmin=0, vmax=1)
ax_see.set_xticks([])
ax_see.set_yticks([])
for s in ax_see.spines.values():
    s.set_edgecolor(GRID)

# Right lower panel: confidence for each digit class.
ax_bar = fig.add_subplot(gs[2, 2])
_style(ax_bar, "Confidence per digit")
bars = ax_bar.bar(range(10), np.zeros(10), color=BLUE, width=0.7)
ax_bar.set_ylim(0, 1.05)
ax_bar.set_xticks(range(10))
ax_bar.grid(axis="x", visible=False)


# -----------------------------------------------------------------------------
# Refresh the display using the latest drawing and the current network state.
# -----------------------------------------------------------------------------
def refresh(full=True):
    """Run preprocessing, classify the sketch, and redraw every panel."""
    small = preprocess(canvas)
    x = small.reshape(1, -1)
    probs = net.predict_proba(x)[0]
    top = probs.argmax()

    # Update the per-digit bar plot.
    for b, p in zip(bars, probs):
        b.set_height(p)
        b.set_color(BLUE)
    if canvas.max() > 0.05:
        bars[top].set_color(GREEN)

    # Update the top prediction and confidence text.
    t_pred.set_text(str(top) if canvas.max() > 0.05 else "-")
    t_conf.set_text(f"{probs[top] * 100:.0f}%" if canvas.max() > 0.05 else "-")

    # Refresh the image panels.
    img.set_data(canvas)
    see.set_data(small)
    if full:
        draw_network(ax_net, net, x, side, "your\ndrawing")

    fig.canvas.draw_idle()


# -----------------------------------------------------------------------------
# Drawing interaction: accumulate strokes on the canvas with a Gaussian brush.
# -----------------------------------------------------------------------------
def stamp(x, y):
    """Add a brush stroke to the canvas, interpolating between last and current point."""
    global canvas
    pts = [(x, y)]
    if state["last_xy"] is not None:
        x0, y0 = state["last_xy"]
        n = int(max(abs(x - x0), abs(y - y0)) / 3) + 1
        pts = [(x0 + (x - x0) * t, y0 + (y - y0) * t) for t in np.linspace(0, 1, n + 1)]

    for px, py in pts:
        canvas = np.maximum(canvas, np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * (BRUSH / 2) ** 2)))
    state["last_xy"] = (x, y)


def clear():
    """Reset the canvas and immediately refresh the visualizations."""
    canvas[:] = 0
    refresh()


# -----------------------------------------------------------------------------
# Mouse event handlers for drawing actions.
# -----------------------------------------------------------------------------
def on_press(e):
    """Start a draw action on left click; right click clears the canvas."""
    if e.button == 3:
        clear()
        return
    if e.button == 1 and e.inaxes is ax_draw and e.xdata is not None:
        state["drawing"] = True
        state["last_xy"] = None
        stamp(e.xdata, e.ydata)
        refresh()


def on_move(e):
    """While dragging, draw continuously and throttle network redraws to keep the UI responsive."""
    if state["drawing"] and e.inaxes is ax_draw and e.xdata is not None:
        stamp(e.xdata, e.ydata)
        now = time.time()
        refresh(full=now - state["last"] > 0.12)
        if now - state["last"] > 0.12:
            state["last"] = now


def on_release(e):
    """Finish a stroke when the mouse is released."""
    if state["drawing"]:
        state["drawing"] = False
        state["last_xy"] = None
        refresh()


# Hook mouse and keyboard events into the figure.
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("key_press_event", lambda e: clear() if e.key == "c" else None)

# Initial render.
refresh()
if __name__ == "__main__":
    plt.show()
