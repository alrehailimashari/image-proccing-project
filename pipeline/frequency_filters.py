"""
pipeline/frequency_filters.py
==============================
Frequency-domain filtering using FFT (Fast Fourier Transform).

Filters implemented:
  - Ideal Low-pass filter
  - Butterworth Low-pass filter
  - Ideal High-pass filter
  - Notch / Band-reject filter (for periodic noise removal)
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Core FFT utilities
# ---------------------------------------------------------------------------

def compute_fft(image):
    """
    Compute the 2D FFT of a grayscale image, shifted so DC is at center.

    Returns
    -------
    tuple of (fft_shifted, magnitude_spectrum)
    """
    f = np.fft.fft2(image.astype(np.float64))
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    return fshift, magnitude


def apply_fft_filter(image, mask):
    """
    Apply a frequency-domain filter mask to an image via FFT.

    Parameters
    ----------
    image : np.ndarray  Grayscale image (uint8).
    mask  : np.ndarray  Frequency-domain mask (float, same shape as image).
                         Values in [0, 1]; 1 = pass, 0 = reject.

    Returns
    -------
    np.ndarray  Filtered image (uint8).
    """
    f = np.fft.fft2(image.astype(np.float64))
    fshift = np.fft.fftshift(f)
    filtered = fshift * mask
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)


def _distance_matrix(shape):
    """Create a matrix of distances from center for filter construction."""
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows).reshape(-1, 1) - crow
    v = np.arange(cols).reshape(1, -1) - ccol
    return np.sqrt(u ** 2 + v ** 2)


# ---------------------------------------------------------------------------
# Low-pass Filters
# ---------------------------------------------------------------------------

def ideal_lowpass(shape, d0=50):
    """
    Ideal low-pass filter mask.

    Passes all frequencies within radius D0 from center, blocks all others.
    Sharp cutoff causes ringing artifacts.

    Parameters
    ----------
    shape : tuple  (rows, cols) of the image.
    d0    : float  Cutoff frequency (radius in pixels).

    Returns
    -------
    np.ndarray  Filter mask (float, shape = image shape).
    """
    D = _distance_matrix(shape)
    mask = np.zeros(shape, dtype=np.float64)
    mask[D <= d0] = 1.0
    return mask


def butterworth_lowpass(shape, d0=50, order=2):
    """
    Butterworth low-pass filter mask.

    Smoother transition than ideal filter, reducing ringing artifacts.
    H(u,v) = 1 / (1 + (D/D0)^(2n))

    Parameters
    ----------
    shape : tuple  (rows, cols) of the image.
    d0    : float  Cutoff frequency.
    order : int    Filter order (higher = sharper cutoff).

    Returns
    -------
    np.ndarray  Filter mask.
    """
    D = _distance_matrix(shape)
    # Avoid division by zero
    D[D == 0] = 1e-10
    mask = 1.0 / (1.0 + (D / d0) ** (2 * order))
    return mask


# ---------------------------------------------------------------------------
# High-pass Filters
# ---------------------------------------------------------------------------

def ideal_highpass(shape, d0=50):
    """
    Ideal high-pass filter mask.

    Blocks all frequencies within radius D0, passes all others.
    Inverse of the ideal low-pass filter.

    Parameters
    ----------
    shape : tuple  (rows, cols).
    d0    : float  Cutoff frequency.

    Returns
    -------
    np.ndarray  Filter mask.
    """
    return 1.0 - ideal_lowpass(shape, d0)


def butterworth_highpass(shape, d0=50, order=2):
    """
    Butterworth high-pass filter mask.

    Parameters
    ----------
    shape : tuple  (rows, cols).
    d0    : float  Cutoff frequency.
    order : int    Filter order.

    Returns
    -------
    np.ndarray  Filter mask.
    """
    return 1.0 - butterworth_lowpass(shape, d0, order)


# ---------------------------------------------------------------------------
# Notch / Band-reject Filter (for periodic noise)
# ---------------------------------------------------------------------------

def notch_reject(shape, center_freq=30, bandwidth=10, direction="horizontal"):
    """
    Notch (band-reject) filter to remove periodic noise.

    Creates notch pairs at the frequency locations corresponding to the
    periodic noise, rejecting a narrow band around those frequencies.

    Parameters
    ----------
    shape        : tuple  (rows, cols).
    center_freq  : float  Frequency of the periodic noise to remove.
    bandwidth    : float  Width of the rejection band (radius around notch center).
    direction    : str    'horizontal' or 'vertical'.

    Returns
    -------
    np.ndarray  Filter mask.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones(shape, dtype=np.float64)

    if direction == "horizontal":
        # Periodic noise along horizontal axis → notch in vertical freq axis
        notch_points = [
            (crow, ccol + center_freq),
            (crow, ccol - center_freq),
        ]
    else:
        notch_points = [
            (crow + center_freq, ccol),
            (crow - center_freq, ccol),
        ]

    for (nr, nc) in notch_points:
        if 0 <= nr < rows and 0 <= nc < cols:
            u = np.arange(rows).reshape(-1, 1) - nr
            v = np.arange(cols).reshape(1, -1) - nc
            D = np.sqrt(u ** 2 + v ** 2)
            mask[D <= bandwidth] = 0.0

    return mask


# ---------------------------------------------------------------------------
# Convenience: apply all frequency filters
# ---------------------------------------------------------------------------

def apply_all_frequency_filters(image, d0_values=None):
    """
    Apply multiple frequency-domain filters and return results.

    Returns dict of {filter_name: filtered_image}.
    """
    if d0_values is None:
        d0_values = [30, 50, 80]

    results = {}
    shape = image.shape

    for d0 in d0_values:
        mask = ideal_lowpass(shape, d0)
        results[f"Ideal LP D0={d0}"] = apply_fft_filter(image, mask)

        mask = butterworth_lowpass(shape, d0, order=2)
        results[f"Butterworth LP D0={d0}"] = apply_fft_filter(image, mask)

    for d0 in d0_values:
        mask = ideal_highpass(shape, d0)
        results[f"Ideal HP D0={d0}"] = apply_fft_filter(image, mask)

    return results


FREQUENCY_FILTERS = {
    "Ideal Low-pass":       ideal_lowpass,
    "Butterworth Low-pass": butterworth_lowpass,
    "Ideal High-pass":      ideal_highpass,
    "Notch Reject":         notch_reject,
}
