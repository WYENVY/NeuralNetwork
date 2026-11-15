"""
augment.py - on-the-fly data augmentation, written in plain NumPy.

Each training image gets a random small rotation, scale, shear and shift, so the
network sees a slightly different version of every digit each epoch. 

Run `python augment.py` to save a preview image of what augmentation does.
"""

# NumPy is the only dependency required for the augmentation math itself.
# We keep the implementation vectorized so batches of images are processed together.
import numpy as np


# ---------------------------------------------------------------------------
# Batch augmentation: apply a random affine transform to every image.
# ---------------------------------------------------------------------------
def augment_batch(X, side, rng, max_rot=15.0, max_shift=0.1, scale=(0.9, 1.1), max_shear=0.1):
    """
    X: (B, side*side) images in [0, 1]. Returns an augmented copy with the same shape.

    The transform is learned per-image and uses an inverse affine map so the
    destination pixels sample from the original image. Bilinear interpolation keeps
    the result smooth even when a source pixel lands between grid points.
    """

    # Number of images in the batch. Each image is flattened into a single vector.
    B = X.shape[0]

    # Reshape each flattened image back into a 2D grid for pixel access.
    imgs = X.reshape(B, side, side)

    # Sample one random transform for each image in the batch.
    # These parameters model the small geometric jitter commonly used in training.
    theta = np.deg2rad(rng.uniform(-max_rot, max_rot, B))
    s = rng.uniform(scale[0], scale[1], B)
    sh = rng.uniform(-max_shear, max_shear, B)
    tx = rng.uniform(-max_shift, max_shift, B) * side
    ty = rng.uniform(-max_shift, max_shift, B) * side

    # Build the affine matrix A for each image.
    # A = R @ Shear @ Scale, and we later need A^-1 to map output pixels back to input pixels.
    cos, sin = np.cos(theta), np.sin(theta)
    A = np.zeros((B, 2, 2))
    A[:, 0, 0] = cos * s
    A[:, 0, 1] = (cos * sh - sin) * s
    A[:, 1, 0] = sin * s
    A[:, 1, 1] = (sin * sh + cos) * s
    Ainv = np.linalg.inv(A)

    # Center the coordinate system so rotation happens around the image midpoint.
    c = (side - 1) / 2.0
    ys, xs = np.mgrid[0:side, 0:side]

    # Destination coordinates are all pixels in the output image, arranged as columns.
    # We subtract the center so the affine transform is applied around the image center.
    dst = np.stack([xs - c, ys - c]).reshape(2, -1)  # (2, side*side), centered

    # The inverse transform tells us where each output pixel came from in the original image.
    # The translation is subtracted after the inverse rotation/scale/shear is applied.
    src = np.einsum("bij,jn->bin", Ainv, dst) \
        - np.einsum("bij,bj->bi", Ainv, np.stack([tx, ty], axis=1))[:, :, None]
    sx, sy = src[:, 0] + c, src[:, 1] + c  # (B, side*side)

    # Bilinear interpolation samples the four nearest pixels around each source coordinate.
    # Values outside the image are treated as zero, which acts like a background.
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    fx, fy = sx - x0, sy - y0

    # Access a pixel while clamping indices to the valid range.
    # The boolean mask keeps out-of-bounds positions from leaking invalid data.
    def at(yi, xi):
        ok = (yi >= 0) & (yi < side) & (xi >= 0) & (xi < side)
        vals = imgs[np.arange(B)[:, None], np.clip(yi, 0, side - 1), np.clip(xi, 0, side - 1)]
        return vals * ok

    # Combine the four neighboring pixels with weights determined by the fractional offset.
    # This produces a smooth image even when the transformed coordinate falls between integer pixels.
    out = (at(y0, x0) * (1 - fx) * (1 - fy)
           + at(y0, x0 + 1) * fx * (1 - fy)
           + at(y0 + 1, x0) * (1 - fx) * fy
           + at(y0 + 1, x0 + 1) * fx * fy)

    # Clamp values back to the input intensity range [0, 1].
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------------------
# Optional demo: render a small preview of the augmentation effect.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # These imports are only needed when the script is run directly for visualization.
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits

    # Load a small set of handwritten digits and normalize them to [0, 1].
    d = load_digits()
    X, side = d.data[:10] / 16.0, 8

    # Use a fixed random seed so the preview is deterministic and easy to compare.
    rng = np.random.default_rng(0)

    # Create a 5x10 grid: 1 original row plus 4 augmented rows.
    fig, axes = plt.subplots(5, 10, figsize=(12, 6.4), facecolor="#0d1117")

    for r in range(5):
        # Keep the first row original; every next row is an augmented copy of the same batch.
        batch = X if r == 0 else augment_batch(X, side, rng)

        for c in range(10):
            # Display each image as a square 8x8 grid with a colored heatmap.
            ax = axes[r, c]
            ax.imshow(batch[c].reshape(side, side), cmap="magma")
            ax.axis("off")

        # Add a short label to the left-most subplot in each row.
        axes[r, 0].set_title("original" if r == 0 else f"augmented #{r}", color="#8b949e",
                             fontsize=9, loc="left")

    # Refine the layout and save the preview to disk for quick inspection.
    fig.tight_layout()
    fig.savefig("augmentation_preview.png", dpi=110, facecolor="#0d1117")
    print("saved augmentation_preview.png")
