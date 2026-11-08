"""
utils.py
========
Helper functions that are not image-processing algorithms themselves:
loading/validating uploads, saving, quality metrics (MSE, PSNR), histogram plots,
and a built-in sample image.
"""

import io
import os

import cv2
import numpy as np
from PIL import Image, ImageOps
from matplotlib.figure import Figure

import image_processing as ip

ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAX_FILE_MB = 25
MAX_SIDE = 800        # larger images are shrunk so the filters stay fast in a live demo


# --------------------------------------------------------------------------
# Loading / saving
# --------------------------------------------------------------------------
def load_image(file_bytes, filename):
    """
    Decode an uploaded file into an RGB uint8 array.
    Returns (rgb_array, info_dict). Raises ValueError with a friendly message if the file is not usable.
    Handles: EXIF rotation, transparency (PNG alpha is composited on white), palette and 16-bit PNGs.
    """
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Please upload a JPG, JPEG or PNG image.")
    if len(file_bytes) == 0:
        raise ValueError("The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        raise ValueError(f"File is larger than {MAX_FILE_MB} MB. Please upload a smaller image.")

    try:
        im = Image.open(io.BytesIO(file_bytes))
        im.load()
    except Exception as exc:       # PIL raises several different error types for broken files
        raise ValueError("This file could not be read as an image. It may be corrupted or not a real image.") from exc

    info = {
        "filename": filename,
        "format": im.format,
        "mode": im.mode,
        "bands": len(im.getbands()),
        "file_size": len(file_bytes),
    }

    im = ImageOps.exif_transpose(im)
    if im.mode.startswith("I") or im.mode == "F":              # 16/32-bit gray: rescale to 8-bit
        a = np.asarray(im, dtype=np.float64)
        top = a.max() if a.max() > 0 else 1.0
        rgb = np.stack([np.rint(a / top * 255).astype(np.uint8)] * 3, axis=-1)
    else:
        if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
            rgba = im.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(background, rgba)
        rgb = np.array(im.convert("RGB"), dtype=np.uint8)

    if rgb.ndim != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("The image has no pixels.")
    return rgb, info


def limit_size(img, max_side=MAX_SIDE):
    """Shrink (INTER_AREA) so the longer side is at most max_side. Returns (image, was_resized)."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, False
    scale = max_side / max(h, w)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), True


def to_png_bytes(img):
    """Encode a uint8 gray/RGB array as PNG bytes (for the download button)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def looks_grayscale(rgb):
    """True if an RGB image has R == G == B everywhere (a gray image saved as colour)."""
    return rgb.ndim == 2 or bool(np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]))


def human_size(n_bytes):
    return f"{n_bytes / 1024:.1f} KB" if n_bytes < 1024 * 1024 else f"{n_bytes / 1024 / 1024:.2f} MB"


def make_sample_image():
    """A synthetic 480x360 test image (gradient, shapes, text) so the app can be demoed without a photo."""
    h, w = 360, 480
    x = np.linspace(40, 200, w, dtype=np.float64)
    img = np.stack([np.tile(x, (h, 1)),
                    np.tile(np.linspace(90, 150, h)[:, None], (1, w)),
                    np.tile(x[::-1], (h, 1))], axis=-1).astype(np.uint8)
    cv2.circle(img, (130, 140), 80, (230, 200, 40), -1)
    cv2.rectangle(img, (260, 60), (440, 200), (20, 60, 160), -1)
    cv2.line(img, (20, 300), (460, 240), (255, 255, 255), 3)
    cv2.putText(img, "IPA 2026", (140, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------
# Quality metrics  (need a REFERENCE image of the same size)
# --------------------------------------------------------------------------
def mse(reference, test):
    """Mean Squared Error = mean( (ref - test)^2 ). float64 is used so uint8 cannot wrap around."""
    if reference.shape != test.shape:
        raise ValueError(f"Images must have the same shape (got {reference.shape} and {test.shape}).")
    diff = reference.astype(np.float64) - test.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(reference, test):
    """PSNR = 10 * log10( 255^2 / MSE ) in dB. Identical images -> infinity."""
    e = mse(reference, test)
    if e == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / e))


# --------------------------------------------------------------------------
# Histogram plots and description
# --------------------------------------------------------------------------
def histogram_figure(img, title="Histogram"):
    """Gray image -> one bar histogram. Colour image -> overlaid R, G, B curves."""
    fig = Figure(figsize=(5, 2.8), dpi=100)
    ax = fig.subplots()
    if img.ndim == 2:
        ax.bar(np.arange(256), ip.channel_histogram(img), width=1.0, color="dimgray")
    else:
        for c, (name, col) in enumerate((("R", "red"), ("G", "green"), ("B", "blue"))):
            ax.plot(ip.channel_histogram(img[..., c]), color=col, label=name, linewidth=1.2)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensity (0 = black, 255 = white)")
    ax.set_ylabel("Pixel count")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def describe_histogram(img):
    """Plain-language rule-of-thumb reading of the gray histogram (a guide, not a measurement of quality)."""
    g = ip.to_gray(img)
    mean, std = float(g.mean()), float(g.std())
    lo, hi = int(g.min()), int(g.max())
    brightness = "dark" if mean < 85 else "bright" if mean > 170 else "medium brightness"
    contrast = "low contrast" if std < 40 else "high contrast" if std > 70 else "moderate contrast"
    text = (f"Mean = {mean:.1f}, standard deviation = {std:.1f}, range = {lo}..{hi}. "
            f"Rule of thumb: this image looks **{brightness}** with **{contrast}**.")
    if lo == hi:
        text += " All pixels have the same value, so the histogram is a single spike."
    return text
