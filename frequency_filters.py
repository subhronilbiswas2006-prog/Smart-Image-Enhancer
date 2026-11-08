"""
frequency_filters.py
====================
Frequency-domain filtering with NumPy's FFT.

Pipeline used (the textbook procedure):
  1. f(x,y) -> optionally zero-pad to P x Q = 2M x 2N (avoids wrap-around error)
  2. F(u,v) = FFT2(f), then fftshift so that the zero frequency (DC) is at the centre
  3. G(u,v) = H(u,v) * F(u,v)          <- H is the filter transfer function
  4. g = real( IFFT2( ifftshift(G) ) ), crop back to M x N
  5. clip / rescale to 0..255 for display

D(u,v) is the distance of frequency point (u,v) from the centre of the shifted spectrum.
D0 is the cutoff frequency, measured in "pixels of the frequency plane".
"""

import numpy as np
from image_processing import ensure_uint8, to_gray, _require_image, MAX_VAL

FAMILIES = ("ideal", "butterworth", "gaussian")
MODES = ("lowpass", "highpass")


def _distance_grid(P, Q):
    """D(u,v) = sqrt(u^2 + v^2) with (0,0) at index (P//2, Q//2), matching np.fft.fftshift."""
    u = np.arange(P) - P // 2
    v = np.arange(Q) - Q // 2
    return np.sqrt(u[:, None] ** 2 + v[None, :] ** 2)


def transfer_function(shape, family, mode, cutoff, order=2):
    """
    Build the centred filter H(u,v) of size `shape` = (P, Q).

      ideal LP        H = 1 if D <= D0 else 0
      Butterworth LP  H = 1 / (1 + (D/D0)^(2n))        (H = 0.5 at D = D0)
      Gaussian LP     H = exp(-D^2 / (2 D0^2))         (H = 0.607 at D = D0)
      High-pass       H_hp = 1 - H_lp
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Cutoff frequency D0 must be greater than 0.")
    if family == "butterworth" and (int(order) != order or order < 1):
        raise ValueError("Butterworth order must be an integer >= 1.")

    D = _distance_grid(*shape)
    if family == "ideal":
        H = (D <= cutoff).astype(np.float64)
    elif family == "butterworth":
        H = 1.0 / (1.0 + (D / cutoff) ** (2 * int(order)))
    else:
        H = np.exp(-(D ** 2) / (2.0 * cutoff ** 2))
    return H if mode == "lowpass" else 1.0 - H


def padded_shape(shape, use_padding=True):
    """Size of the frequency plane: 2M x 2N when padding is on, otherwise M x N."""
    M, N = shape[:2]
    return (2 * M, 2 * N) if use_padding else (M, N)


def fft_spectrum(img):
    """Centred log-magnitude spectrum  log(1 + |F|), scaled to 0..255 for display (gray version of img)."""
    gray = to_gray(img).astype(np.float64)
    F = np.fft.fftshift(np.fft.fft2(gray))
    s = np.log1p(np.abs(F))
    s = s - s.min()
    if s.max() == 0:
        return np.zeros(s.shape, dtype=np.uint8)
    return ensure_uint8(s / s.max() * MAX_VAL)


def reconstruction_error(img):
    """Largest absolute difference between a gray image and IFFT2(FFT2(image)). Should be ~1e-12."""
    f = to_gray(img).astype(np.float64)
    return float(np.max(np.abs(np.fft.ifft2(np.fft.fft2(f)).real - f)))


def _filter_channel(ch, family, mode, cutoff, order, use_padding):
    f = ch.astype(np.float64)
    M, N = f.shape
    P, Q = padded_shape((M, N), use_padding)
    padded = np.zeros((P, Q))
    padded[:M, :N] = f
    F = np.fft.fftshift(np.fft.fft2(padded))
    H = transfer_function((P, Q), family, mode, cutoff, order)
    g = np.fft.ifft2(np.fft.ifftshift(F * H)).real     # imaginary part is only rounding error
    return g[:M, :N]


def frequency_filter_float(img, family, mode, cutoff, order=2, use_padding=True):
    """Filtered image as raw float64 (not yet clipped). Colour images are filtered channel by channel."""
    _require_image(img)
    if img.ndim == 2:
        return _filter_channel(img, family, mode, cutoff, order, use_padding)
    return np.stack([_filter_channel(img[..., c], family, mode, cutoff, order, use_padding)
                     for c in range(img.shape[2])], axis=-1)


def to_display(arr, display="clip"):
    """
    display = "clip"      : round and clip to 0..255 (right for low-pass results)
    display = "normalize" : min-max scale the whole array to 0..255 (needed for high-pass results,
                            because removing the DC term makes the mean ~0 and most values negative)
    """
    if display == "clip":
        return ensure_uint8(arr)
    if display == "normalize":
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-9:
            return np.zeros(arr.shape, dtype=np.uint8)
        return ensure_uint8((arr - lo) / (hi - lo) * MAX_VAL)
    raise ValueError("display must be 'clip' or 'normalize'")


def apply_frequency_filter(img, family, mode, cutoff, order=2, use_padding=True, display=None):
    """Filter and convert to a displayable uint8 image. Default display: clip for low-pass, normalize for high-pass."""
    if display is None:
        display = "clip" if mode == "lowpass" else "normalize"
    return to_display(frequency_filter_float(img, family, mode, cutoff, order, use_padding), display)


def mask_to_image(H):
    """Show a filter H (values 0..1) as a gray image."""
    return ensure_uint8(H * MAX_VAL)
