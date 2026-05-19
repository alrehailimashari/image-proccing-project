"""
pipeline/comparator.py
======================
Image quality comparison module.

Computes quantitative metrics (MSE, PSNR) and generates visual
comparison figures (original vs noisy vs restored vs difference map).
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os


def compute_mse(original, restored):
    """
    Compute Mean Squared Error between two images.

    MSE = (1/N) * Σ (original - restored)²

    Parameters
    ----------
    original : np.ndarray  Reference image (uint8).
    restored : np.ndarray  Processed image (uint8).

    Returns
    -------
    float  MSE value (lower is better, 0 = identical).
    """
    return np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)


def compute_psnr(original, restored):
    """
    Compute Peak Signal-to-Noise Ratio.

    PSNR = 10 * log10(MAX² / MSE)
    where MAX = 255 for 8-bit images.

    Parameters
    ----------
    original : np.ndarray  Reference image.
    restored : np.ndarray  Processed image.

    Returns
    -------
    float  PSNR in dB (higher is better). Returns inf if MSE = 0.
    """
    mse = compute_mse(original, restored)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_difference_map(original, enhanced):
    """
    Compute the absolute difference map between original and enhanced images.

    Parameters
    ----------
    original : np.ndarray  Reference image.
    enhanced : np.ndarray  Processed image.

    Returns
    -------
    np.ndarray  Absolute difference (uint8), scaled for visibility.
    """
    diff = np.abs(original.astype(np.float64) - enhanced.astype(np.float64))
    # Scale to full range for visibility
    if diff.max() > 0:
        diff = (diff / diff.max() * 255).astype(np.uint8)
    else:
        diff = diff.astype(np.uint8)
    return diff


def generate_comparison_figure(original, noisy, restored, diff_map,
                                mse_val, psnr_val, object_name, noise_type,
                                best_filter_name, save_path=None):
    """
    Generate a 4-panel comparison figure.

    Layout:
      [Original] [Noisy] [Restored] [Difference Map]
      Caption with MSE, PSNR, and best filter name.

    Parameters
    ----------
    save_path : str or None
        If provided, save figure to this path. Otherwise return fig.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f"{object_name} — Noise: {noise_type} | Best Filter: {best_filter_name}",
                 fontsize=14, fontweight='bold')

    titles = ["Original", "Noisy", f"Restored ({best_filter_name})", "Difference Map"]
    images = [original, noisy, restored, diff_map]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # Add metrics text
    fig.text(0.5, 0.02,
             f"MSE = {mse_val:.2f}   |   PSNR = {psnr_val:.2f} dB",
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path
    return fig


def evaluate_all_filters(original, noisy, filter_results):
    """
    Evaluate all filter results against the original and return a ranked table.

    Parameters
    ----------
    original : np.ndarray
        Clean original image.
    noisy : np.ndarray
        Noisy image (for reference MSE/PSNR).
    filter_results : dict
        {filter_name: filtered_image}

    Returns
    -------
    list of dict
        Sorted by PSNR descending. Each dict has:
        'filter', 'mse', 'psnr'.
    """
    results = []

    # Noisy baseline
    results.append({
        "filter": "Noisy (no filter)",
        "mse": compute_mse(original, noisy),
        "psnr": compute_psnr(original, noisy),
    })

    for name, img in filter_results.items():
        results.append({
            "filter": name,
            "mse": compute_mse(original, img),
            "psnr": compute_psnr(original, img),
        })

    # Sort by PSNR descending (best first)
    results.sort(key=lambda x: x["psnr"], reverse=True)
    return results
