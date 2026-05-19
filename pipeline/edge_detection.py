"""
pipeline/edge_detection.py
==========================
Edge detection and sharpening operators.

IMPORTANT DISTINCTION:
  - These are NOT noise-removal filters.
  - Sobel detects edges (gradient magnitude).
  - Laplacian detects edges / sharpens via second derivative.
  - Laplacian amplifies noise if applied before denoising.
  - Always apply denoising FIRST, then use these for edge analysis.
"""

import cv2
import numpy as np


def sobel_edges(image, ksize=3):
    """
    Compute edge magnitude using the Sobel operator.

    Computes horizontal (dx) and vertical (dy) gradients separately,
    then combines them into a magnitude image.

    Parameters
    ----------
    image : np.ndarray  Grayscale image (uint8).
    ksize : int         Sobel kernel size (1, 3, 5, or 7).

    Returns
    -------
    tuple of (magnitude, grad_x, grad_y)
        magnitude : np.ndarray  Edge magnitude image (uint8).
        grad_x    : np.ndarray  Horizontal gradient (uint8).
        grad_y    : np.ndarray  Vertical gradient (uint8).
    """
    # Compute gradients in float to avoid overflow
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)

    # Magnitude
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    grad_x_abs = cv2.convertScaleAbs(grad_x)
    grad_y_abs = cv2.convertScaleAbs(grad_y)

    return magnitude, grad_x_abs, grad_y_abs


def laplacian_edges(image, ksize=3):
    """
    Apply the Laplacian operator for edge detection / sharpening.

    The Laplacian computes the second derivative, highlighting regions
    of rapid intensity change. It is sensitive to noise and should only
    be applied AFTER denoising.

    Parameters
    ----------
    image : np.ndarray  Grayscale image (uint8).
    ksize : int         Kernel size for the Laplacian (must be odd).

    Returns
    -------
    np.ndarray  Laplacian edge image (uint8, absolute values).
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    return cv2.convertScaleAbs(laplacian)


def sharpen_with_laplacian(image, alpha=1.0, ksize=3):
    """
    Sharpen an image using the Laplacian operator.

    Sharpened = Original - alpha * Laplacian

    Parameters
    ----------
    image : np.ndarray  Grayscale image (uint8).
    alpha : float       Sharpening strength.
    ksize : int         Kernel size.

    Returns
    -------
    np.ndarray  Sharpened image (uint8).
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    sharpened = image.astype(np.float64) - alpha * laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)
