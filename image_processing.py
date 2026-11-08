"""
image_processing.py
===================
Spatial-domain image processing algorithms for the Smart Image Enhancer.

Conventions used everywhere in this file
----------------------------------------
* Images are NumPy arrays of dtype uint8 (values 0..255).
* Grayscale image -> shape (H, W).  Colour image -> shape (H, W, 3) in RGB order.
* Every function RETURNS A NEW ARRAY. The input image is never modified, so the
  original image stays safe (this is how the "Reset" feature works).
* Calculations are done in float64 and converted back with `ensure_uint8`
  (round + clip to 0..255), so uint8 overflow / underflow cannot happen.
* Border handling for neighbourhood filters: "reflect" padding
  (same as OpenCV's BORDER_REFLECT_101).

Each function says whether it is a real implementation of the mathematics
("implemented from the formula") or uses an OpenCV shortcut ("library call").
"""

import numpy as np
import cv2
from numpy.lib.stride_tricks import sliding_window_view

L = 256            # number of gray levels (8-bit image)
MAX_VAL = L - 1    # 255


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def ensure_uint8(arr):
    """Round to nearest integer and clip to 0..255, then cast to uint8."""
    return np.clip(np.rint(arr), 0, MAX_VAL).astype(np.uint8)


def _require_image(img):
    """Check that `img` is a uint8 image of shape (H,W) or (H,W,3)."""
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError("Image must be a uint8 NumPy array.")
    if img.ndim not in (2, 3) or img.size == 0:
        raise ValueError("Image must have shape (H, W) or (H, W, C) and be non-empty.")


def _per_channel(img, func):
    """Apply a 2-D function to a gray image, or to each channel of a colour image."""
    _require_image(img)
    if img.ndim == 2:
        return func(img)
    return np.stack([func(img[..., c]) for c in range(img.shape[2])], axis=-1)


def is_constant(img):
    """True if every pixel has the same value (histogram is a single spike)."""
    return bool(img.min() == img.max())


def to_gray(img):
    """RGB -> gray using Y = 0.299 R + 0.587 G + 0.114 B (library call: cv2.cvtColor)."""
    _require_image(img)
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def scale_abs_to_uint8(arr):
    """
    Display scaling for signed results (gradients, Laplacian):
    take |arr| and stretch so the largest value becomes 255. All-zero -> black.
    """
    a = np.abs(np.asarray(arr, dtype=np.float64))
    m = a.max()
    if m == 0:
        return np.zeros(a.shape, dtype=np.uint8)
    return ensure_uint8(a / m * MAX_VAL)


def channel_histogram(channel):
    """Histogram h(r_k) = number of pixels with intensity k, for k = 0..255."""
    return np.bincount(channel.ravel(), minlength=L)


# --------------------------------------------------------------------------
# A. Point (intensity) transformations   s = T(r)
# Each one builds a 256-entry lookup table T and then does  out = T[img].
# --------------------------------------------------------------------------
def _apply_table(img, table):
    _require_image(img)
    return table[img]          # NumPy fancy indexing: works for gray and colour


def negative(img):
    """Image negative:  s = (L - 1) - r   (implemented from the formula)."""
    r = np.arange(L, dtype=np.float64)
    return _apply_table(img, ensure_uint8(MAX_VAL - r))


def log_transform(img, c_scale=1.0):
    """
    Log transform:  s = c * log(1 + r)
    c is chosen so that r = 255 maps to 255:  c = 255 / log(256).
    `c_scale` multiplies that c (1.0 = full-range mapping). Implemented from the formula.
    """
    if not np.isfinite(c_scale) or c_scale <= 0:
        raise ValueError("c_scale must be a positive number.")
    r = np.arange(L, dtype=np.float64)
    c = MAX_VAL / np.log(L)
    return _apply_table(img, ensure_uint8(c_scale * c * np.log1p(r)))


def gamma_transform(img, gamma):
    """
    Power-law (gamma) transform:  s = (L-1) * (r/(L-1)) ** gamma
    r is normalised to 0..1 first. gamma < 1 brightens, gamma > 1 darkens.
    Implemented from the formula.
    """
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be a positive number (> 0).")
    r = np.arange(L, dtype=np.float64) / MAX_VAL
    return _apply_table(img, ensure_uint8(MAX_VAL * np.power(r, gamma)))


def contrast_stretch(img, low_pct=0.0, high_pct=100.0):
    """
    Contrast stretching (min-max / percentile):
        s = (r - r_min) / (r_max - r_min) * 255,  values outside [r_min, r_max] are clipped.
    r_min, r_max are the `low_pct` and `high_pct` percentiles of the image
    (0 and 100 = true min and max). For colour images ONE common range is used for
    all channels so colours are not shifted. Constant image -> returned unchanged
    (r_max == r_min would be a division by zero).
    """
    _require_image(img)
    if not (0 <= low_pct < high_pct <= 100):
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100.")
    r_min, r_max = np.percentile(img, [low_pct, high_pct])
    if r_max <= r_min:
        return img.copy()
    r = np.arange(L, dtype=np.float64)
    table = ensure_uint8((r - r_min) / (r_max - r_min) * MAX_VAL)
    return _apply_table(img, table)


def _equalize_channel(ch):
    """Histogram equalisation of one gray channel using the CDF (implemented from the formula)."""
    hist = np.bincount(ch.ravel(), minlength=L)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0][0]              # CDF value at the darkest occurring level
    total = ch.size
    if total == cdf_min:                   # constant image -> nothing to equalise
        return ch.copy()
    # s_k = round( (cdf(k) - cdf_min) / (N - cdf_min) * 255 )
    table = np.rint((cdf - cdf_min) / (total - cdf_min) * MAX_VAL)
    table = np.clip(table, 0, MAX_VAL).astype(np.uint8)
    return table[ch]


def histogram_equalization(img):
    """
    Histogram equalisation. Gray image: equalise directly.
    Colour image: convert to YCrCb, equalise ONLY the luminance Y, convert back
    (equalising R,G,B separately would change the colours).
    """
    _require_image(img)
    if img.ndim == 2:
        return _equalize_channel(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = _equalize_channel(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def gray_level_slicing(img, low, high, highlight=255, preserve_background=True):
    """
    Gray-level slicing on the intensity range [low, high].
      preserve_background=True : pixels inside the range -> `highlight`, others unchanged
      preserve_background=False: pixels inside -> `highlight`, all others -> 0 (binary-like)
    """
    gray = to_gray(img)
    for name, v in (("low", low), ("high", high), ("highlight", highlight)):
        if not (0 <= v <= MAX_VAL):
            raise ValueError(f"{name} must be between 0 and 255.")
    if low > high:
        raise ValueError("Lower limit must not be greater than the upper limit.")
    mask = (gray >= low) & (gray <= high)
    out = gray.copy() if preserve_background else np.zeros_like(gray)
    out[mask] = np.uint8(highlight)
    return out


def bit_plane_slice(img, bit):
    """
    Bit plane k of a gray image:  plane = (pixel >> k) & 1, shown as 0 or 255.
    bit 7 = most significant (shape of the image), bit 0 = least significant (almost noise).
    """
    if not isinstance(bit, (int, np.integer)) or not (0 <= bit <= 7):
        raise ValueError("Bit plane must be an integer from 0 to 7.")
    gray = to_gray(img)
    return (((gray >> bit) & 1) * MAX_VAL).astype(np.uint8)


def bit_plane_reconstruct(img, planes):
    """Rebuild an image using only the chosen bit planes: sum of ((pixel >> k) & 1) * 2^k."""
    gray = to_gray(img)
    out = np.zeros(gray.shape, dtype=np.uint16)
    for k in planes:
        if not (0 <= k <= 7):
            raise ValueError("Bit planes must be between 0 and 7.")
        out += (((gray >> k) & 1).astype(np.uint16) << k)
    return out.astype(np.uint8)


def add_salt_and_pepper(img, amount=0.05, salt_ratio=0.5, seed=None):
    """
    Simulated impulse noise. `amount` = fraction of pixels corrupted (0..1);
    of those, `salt_ratio` become 255 (salt) and the rest 0 (pepper).
    A pixel is corrupted in all its channels. `seed` makes the result repeatable.
    """
    _require_image(img)
    if not (0 <= amount <= 1) or not (0 <= salt_ratio <= 1):
        raise ValueError("amount and salt_ratio must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    p = rng.random(img.shape[:2])
    out = img.copy()
    out[p < amount * salt_ratio] = 255                        # salt
    out[(p >= amount * salt_ratio) & (p < amount)] = 0        # pepper
    return out


# --------------------------------------------------------------------------
# B. Spatial filtering (neighbourhood operations)
# --------------------------------------------------------------------------
def _validate_kernel(k):
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 3 or k % 2 == 0:
        raise ValueError("Kernel size must be an odd integer >= 3 (e.g. 3, 5, 7).")


def _windows(channel, k):
    """
    Return a view of shape (H, W, k, k): for every pixel, the k x k neighbourhood
    around it. The image is padded by k//2 pixels using reflection first, so the
    output has the same size as the input.
    """
    padded = np.pad(channel, k // 2, mode="reflect")
    return sliding_window_view(padded, (k, k))


def mean_filter(img, k=3):
    """Arithmetic mean filter: average of the k x k neighbourhood. (implemented from the formula)"""
    _validate_kernel(k)
    return _per_channel(img, lambda ch: ensure_uint8(_windows(ch, k).mean(axis=(-2, -1))))


def median_filter(img, k=3):
    """Median filter: middle value of the sorted k x k neighbourhood. Best against salt-and-pepper noise."""
    _validate_kernel(k)
    return _per_channel(img, lambda ch: ensure_uint8(np.median(_windows(ch, k), axis=(-2, -1))))


def max_filter(img, k=3):
    """Maximum filter: largest value in the neighbourhood (removes pepper, brightens/dilates)."""
    _validate_kernel(k)
    return _per_channel(img, lambda ch: _windows(ch, k).max(axis=(-2, -1)))


def min_filter(img, k=3):
    """Minimum filter: smallest value in the neighbourhood (removes salt, darkens/erodes)."""
    _validate_kernel(k)
    return _per_channel(img, lambda ch: _windows(ch, k).min(axis=(-2, -1)))


def midpoint_filter(img, k=3):
    """Midpoint filter: (max + min) / 2 of the neighbourhood."""
    _validate_kernel(k)

    def f(ch):
        w = _windows(ch, k)
        return ensure_uint8((w.max(axis=(-2, -1)).astype(np.float64) + w.min(axis=(-2, -1))) / 2.0)

    return _per_channel(img, f)


def max_alpha_trim(k):
    """Largest allowed d for a k x k kernel (k*k - 1; it leaves just the median)."""
    _validate_kernel(k)
    return k * k - 1


def alpha_trimmed_mean_filter(img, k=3, d=2):
    """
    Alpha-trimmed mean filter. In every k x k neighbourhood:
      1. sort the k*k values,
      2. delete the d/2 smallest and the d/2 largest,
      3. average the remaining k*k - d values.
    d = 0  -> arithmetic mean filter;   d = k*k-1 -> median filter.
    Valid d: even integer, 0 <= d <= k*k - 1.
    """
    _validate_kernel(k)
    if isinstance(d, bool) or not isinstance(d, (int, np.integer)):
        raise ValueError("d must be an integer.")
    if d < 0 or d > k * k - 1:
        raise ValueError(f"For a {k}x{k} kernel, d must be between 0 and {k * k - 1}.")
    if d % 2 != 0:
        raise ValueError("d must be even, because d/2 lowest and d/2 highest values are removed.")
    half = d // 2

    def f(ch):
        h, w = ch.shape
        srt = np.sort(_windows(ch, k).reshape(h, w, k * k), axis=-1)
        return ensure_uint8(srt[..., half:k * k - half].mean(axis=-1))

    return _per_channel(img, f)


def apply_kernel(channel, kernel):
    """
    Slide `kernel` over a 2-D array (reflect padding), returning float64 output the
    same size as the input. The kernel is placed directly on the neighbourhood
    (correlation, the form used in textbooks for Sobel / Laplacian masks).
    For symmetric kernels this equals true convolution.
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd.")
    padded = np.pad(channel.astype(np.float64), ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="reflect")
    return np.einsum("hwij,ij->hw", sliding_window_view(padded, (kh, kw)), kernel)


def weighted_mean_kernel(k=3):
    """
    Weighted-mean mask: binomial weights, centre pixel counts most.
    3x3 = [[1,2,1],[2,4,2],[1,2,1]]/16 ; 5x5 uses the row [1,4,6,4,1]. Weights sum to 1.
    """
    _validate_kernel(k)
    from math import comb
    row = np.array([comb(k - 1, i) for i in range(k)], dtype=np.float64)
    kernel = np.outer(row, row)
    return kernel / kernel.sum()


def weighted_mean_filter(img, k=3):
    """Weighted mean filter (approximates Gaussian smoothing)."""
    kernel = weighted_mean_kernel(k)
    return _per_channel(img, lambda ch: ensure_uint8(apply_kernel(ch, kernel)))


SPATIAL_FILTERS = [
    "Arithmetic mean", "Median", "Maximum", "Minimum",
    "Midpoint", "Alpha-trimmed mean", "Weighted mean",
]


def apply_named_filter(img, name, k=3, d=2):
    """Run one of SPATIAL_FILTERS by name (used by the app to avoid repeated if/else code)."""
    table = {
        "Arithmetic mean": lambda: mean_filter(img, k),
        "Median": lambda: median_filter(img, k),
        "Maximum": lambda: max_filter(img, k),
        "Minimum": lambda: min_filter(img, k),
        "Midpoint": lambda: midpoint_filter(img, k),
        "Alpha-trimmed mean": lambda: alpha_trimmed_mean_filter(img, k, d),
        "Weighted mean": lambda: weighted_mean_filter(img, k),
    }
    if name not in table:
        raise ValueError(f"Unknown filter: {name}")
    return table[name]()


# --------------------------------------------------------------------------
# C. Edge detection and sharpening
# --------------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)   # responds to left->right intensity change
SOBEL_Y = SOBEL_X.T.copy()                            # [[-1,-2,-1],[0,0,0],[1,2,1]]: top->bottom change
LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
LAPLACIAN_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)


def sobel_gradients(img):
    """Return (Gx, Gy) as float64 arrays computed on the gray version of the image."""
    gray = to_gray(img)
    return apply_kernel(gray, SOBEL_X), apply_kernel(gray, SOBEL_Y)


def sobel_x(img):
    """Sobel Gx (horizontal gradient -> highlights VERTICAL edges). |Gx| scaled to 0..255 for display."""
    return scale_abs_to_uint8(sobel_gradients(img)[0])


def sobel_y(img):
    """Sobel Gy (vertical gradient -> highlights HORIZONTAL edges). |Gy| scaled to 0..255 for display."""
    return scale_abs_to_uint8(sobel_gradients(img)[1])


def sobel_magnitude(img):
    """Gradient magnitude  sqrt(Gx^2 + Gy^2), scaled to 0..255 for display."""
    gx, gy = sobel_gradients(img)
    return scale_abs_to_uint8(np.hypot(gx, gy))


def laplacian_response(img, include_diagonals=False):
    """Raw (signed, float64) Laplacian of the gray image."""
    kernel = LAPLACIAN_8 if include_diagonals else LAPLACIAN_4
    return apply_kernel(to_gray(img), kernel)


def laplacian_edges(img, include_diagonals=False):
    """Laplacian edge map, |response| scaled to 0..255 for display."""
    return scale_abs_to_uint8(laplacian_response(img, include_diagonals))


def laplacian_sharpen(img, strength=1.0, include_diagonals=False):
    """
    Laplacian sharpening:  g = f - strength * Laplacian(f)
    (minus sign because the kernel centre is negative). Works on gray or colour.
    """
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Strength must be >= 0.")
    kernel = LAPLACIAN_8 if include_diagonals else LAPLACIAN_4
    return _per_channel(img, lambda ch: ensure_uint8(ch.astype(np.float64) - strength * apply_kernel(ch, kernel)))


def _sharpen_with_mask(img, sigma, k):
    """g = f + k * (f - blurred(f)); the Gaussian blur is a library call (cv2.GaussianBlur)."""
    _require_image(img)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Sigma must be positive.")
    f = img.astype(np.float32)
    blurred = cv2.GaussianBlur(f, (0, 0), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT_101)
    mask = f - blurred                       # the "unsharp mask": fine detail only
    return ensure_uint8(f + k * mask)


def unsharp_mask(img, sigma=1.5, amount=1.0):
    """Unsharp masking: g = f + 1*(f - blur(f)) when amount = 1 (amount can be 0..1+)."""
    if amount < 0:
        raise ValueError("Amount must be >= 0.")
    return _sharpen_with_mask(img, sigma, amount)


def high_boost(img, sigma=1.5, k=2.0):
    """High-boost filtering: g = f + k*(f - blur(f)) with k > 1 (k = 1 would be plain unsharp masking)."""
    if k <= 1:
        raise ValueError("High-boost factor k must be greater than 1 (k = 1 is unsharp masking).")
    return _sharpen_with_mask(img, sigma, k)
