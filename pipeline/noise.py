"""
pipeline/noise.py
=================
Noise injection module.
Applies controlled noise models to clean grayscale images.

Supported noise types:
  - Salt & Pepper (impulse noise)
  - Gaussian (additive white Gaussian noise)
  - Periodic (additive sinusoidal pattern)
  - Gamma / Erlang (additive gamma-distributed noise — NOT speckle)
  - Uniform (additive uniform-distributed noise)
"""

import numpy as np
import cv2


def add_salt_pepper(image, density=0.05):
    """
    Add salt-and-pepper (impulse) noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (uint8).
    density : float
        Fraction of pixels affected (0.0–1.0).

    Returns
    -------
    np.ndarray
        Noisy image (uint8).
    """
    noisy = image.copy()
    total_pixels = image.size
    num_salt = int(total_pixels * density / 2)
    num_pepper = int(total_pixels * density / 2)

    # Salt (white pixels)
    coords_salt = [np.random.randint(0, dim, num_salt) for dim in image.shape]
    noisy[coords_salt[0], coords_salt[1]] = 255

    # Pepper (black pixels)
    coords_pepper = [np.random.randint(0, dim, num_pepper) for dim in image.shape]
    noisy[coords_pepper[0], coords_pepper[1]] = 0

    return noisy


def add_gaussian(image, mean=0, sigma=25):
    """
    Add additive white Gaussian noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (uint8).
    mean : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    np.ndarray
        Noisy image (uint8).
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_periodic(image, frequency=30, amplitude=40, direction="horizontal"):
    """
    Add periodic (sinusoidal) noise.

    This adds a sine-wave pattern across the image, which can be removed
    using notch or band-reject filtering in the frequency domain.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (uint8).
    frequency : float
        Spatial frequency of the sinusoidal pattern (cycles across image).
    amplitude : float
        Peak amplitude of the sine wave.
    direction : str
        'horizontal' or 'vertical' direction of the wave.

    Returns
    -------
    np.ndarray
        Noisy image (uint8).
    """
    rows, cols = image.shape
    if direction == "horizontal":
        x = np.arange(cols)
        pattern = amplitude * np.sin(2 * np.pi * frequency * x / cols)
        noise = np.tile(pattern, (rows, 1))
    else:
        y = np.arange(rows)
        pattern = amplitude * np.sin(2 * np.pi * frequency * y / rows)
        noise = np.tile(pattern.reshape(-1, 1), (1, cols))

    noisy = image.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_gamma(image, shape=5.0, scale=5.0):
    """
    Add Gamma/Erlang-distributed ADDITIVE noise.

    NOTE: This is NOT speckle noise. Speckle is multiplicative (img * noise).
    This applies additive gamma-distributed noise: output = img + gamma_noise.
    The gamma distribution is shifted to have zero mean so it doesn't
    systematically brighten the image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (uint8).
    shape : float
        Shape parameter (k) of the gamma distribution.
    scale : float
        Scale parameter (θ) of the gamma distribution.

    Returns
    -------
    np.ndarray
        Noisy image (uint8).
    """
    gamma_noise = np.random.gamma(shape, scale, image.shape)
    # Shift to zero mean so noise is centered
    gamma_noise = gamma_noise - (shape * scale)
    noisy = image.astype(np.float64) + gamma_noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_uniform(image, low=-30, high=30):
    """
    Add additive uniform-distributed noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale input image (uint8).
    low : float
        Lower bound of the uniform distribution.
    high : float
        Upper bound of the uniform distribution.

    Returns
    -------
    np.ndarray
        Noisy image (uint8).
    """
    noise = np.random.uniform(low, high, image.shape)
    noisy = image.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


# ---- Convenience lookup ----
NOISE_FUNCTIONS = {
    "salt_pepper": add_salt_pepper,
    "gaussian":    add_gaussian,
    "periodic":    add_periodic,
    "gamma":       add_gamma,
    "uniform":     add_uniform,
}

# Default noise assignments per object
DEFAULT_NOISE_CONFIG = {
    "circle":  {"type": "salt_pepper", "params": {"density": 0.05}},
    "wrench":  {"type": "gaussian",    "params": {"mean": 0, "sigma": 25}},
    "hexagon": {"type": "periodic",    "params": {"frequency": 30, "amplitude": 40}},
    "leaf":    {"type": "gamma",       "params": {"shape": 5.0, "scale": 5.0}},
    "washer":  {"type": "uniform",     "params": {"low": -30, "high": 30}},
}


def apply_noise(image, noise_type, **params):
    """
    Apply a specific noise type to an image.

    Parameters
    ----------
    image : np.ndarray
        Clean grayscale image.
    noise_type : str
        One of: 'salt_pepper', 'gaussian', 'periodic', 'gamma', 'uniform'.
    **params
        Keyword arguments passed to the noise function.

    Returns
    -------
    np.ndarray
        Noisy image.
    """
    if noise_type not in NOISE_FUNCTIONS:
        raise ValueError(f"Unknown noise type: {noise_type}. "
                         f"Choose from {list(NOISE_FUNCTIONS.keys())}")
    return NOISE_FUNCTIONS[noise_type](image, **params)
