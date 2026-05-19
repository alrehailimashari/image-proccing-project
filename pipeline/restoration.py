"""
pipeline/restoration.py
=======================
Spatial-domain filters for noise reduction and image enhancement.

Contains ONLY smoothing / noise-reduction filters:
  - Mean filter (box blur)
  - Gaussian filter
  - Median filter
  - Histogram equalization
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)

NOTE: Sobel and Laplacian are edge-detection / sharpening operators,
      NOT noise-removal filters. They are in edge_detection.py.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Smoothing Filters
# ---------------------------------------------------------------------------

def mean_filter(image, ksize=5):
    """
    Apply mean (box) filter for noise reduction.

    Replaces each pixel with the average of its neighbors.
    Effective for Gaussian and uniform noise, but blurs edges.

    Parameters
    ----------
    image : np.ndarray   Grayscale image (uint8).
    ksize : int          Kernel size (must be odd).

    Returns
    -------
    np.ndarray  Filtered image.
    """
    return cv2.blur(image, (ksize, ksize))


def gaussian_filter(image, ksize=5, sigma=1.0):
    """
    Apply Gaussian filter for noise reduction.

    Uses a weighted average with Gaussian kernel, giving more weight
    to the center pixel. Better edge preservation than mean filter.

    Parameters
    ----------
    image : np.ndarray   Grayscale image (uint8).
    ksize : int          Kernel size (must be odd).
    sigma : float        Standard deviation of the Gaussian kernel.

    Returns
    -------
    np.ndarray  Filtered image.
    """
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def median_filter(image, ksize=5):
    """
    Apply median filter for noise reduction.

    Replaces each pixel with the median of its neighborhood.
    Excellent for salt-and-pepper (impulse) noise because it
    preserves edges while removing outlier pixels.

    Parameters
    ----------
    image : np.ndarray   Grayscale image (uint8).
    ksize : int          Kernel size (must be odd).

    Returns
    -------
    np.ndarray  Filtered image.
    """
    return cv2.medianBlur(image, ksize)


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------

def histogram_equalization(image):
    """
    Apply global histogram equalization.

    Spreads pixel intensities across the full range [0, 255],
    improving contrast. May over-amplify noise in some cases.

    Parameters
    ----------
    image : np.ndarray  Grayscale image (uint8).

    Returns
    -------
    np.ndarray  Enhanced image.
    """
    return cv2.equalizeHist(image)


def clahe(image, clip_limit=2.0, tile_size=8):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Unlike global histogram equalization, CLAHE divides the image into
    tiles and applies equalization locally, preventing over-amplification
    of noise while improving local contrast.

    Parameters
    ----------
    image : np.ndarray  Grayscale image (uint8).
    clip_limit : float  Threshold for contrast limiting.
    tile_size : int     Size of each tile (grid dimension).

    Returns
    -------
    np.ndarray  Enhanced image.
    """
    clahe_obj = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )
    return clahe_obj.apply(image)


# ---------------------------------------------------------------------------
# Batch: apply all spatial filters and return results
# ---------------------------------------------------------------------------

def apply_all_spatial_filters(image):
    """
    Apply all spatial filters with multiple parameter settings.

    Returns a dict of {filter_name: filtered_image}.
    """
    results = {}

    # Mean filter with different kernel sizes
    for k in [3, 5, 7]:
        results[f"Mean {k}x{k}"] = mean_filter(image, ksize=k)

    # Gaussian filter with different parameters
    for k, s in [(3, 1.0), (5, 1.0), (5, 1.5), (5, 2.0), (7, 2.0)]:
        results[f"Gaussian {k}x{k} σ={s}"] = gaussian_filter(image, ksize=k, sigma=s)

    # Median filter with different kernel sizes
    for k in [3, 5, 7]:
        results[f"Median {k}x{k}"] = median_filter(image, ksize=k)

    return results


SPATIAL_FILTERS = {
    "Mean":     mean_filter,
    "Gaussian": gaussian_filter,
    "Median":   median_filter,
}
