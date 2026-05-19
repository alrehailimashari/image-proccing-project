"""
pipeline/segmentation.py
========================
Image segmentation using thresholding methods.

Methods implemented:
  - Global Thresholding (fixed threshold)
  - Otsu's Thresholding (automatic optimal threshold)
  - Adaptive Thresholding (local neighborhood-based)

Evaluation:
  - IoU (Intersection over Union) against ground-truth when available
  - Qualitative criteria when ground-truth is unavailable
"""

import cv2
import numpy as np


def global_threshold(image, thresh_value=127):
    """
    Apply global (fixed) thresholding.

    Pixels above thresh_value become 255 (white), others become 0 (black).

    Parameters
    ----------
    image : np.ndarray      Grayscale image (uint8).
    thresh_value : int       Threshold value (0–255).

    Returns
    -------
    np.ndarray  Binary mask (uint8, values 0 or 255).
    """
    _, binary = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    return binary


def otsu_threshold(image):
    """
    Apply Otsu's automatic thresholding.

    Otsu's method finds the optimal threshold by minimizing intra-class
    variance. Best for bimodal histograms (clear object/background separation).

    Parameters
    ----------
    image : np.ndarray  Grayscale image (uint8).

    Returns
    -------
    tuple of (binary_mask, optimal_threshold)
        binary_mask : np.ndarray  Binary mask (uint8).
        optimal_threshold : float  The auto-computed threshold value.
    """
    optimal_thresh, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary, optimal_thresh


def adaptive_threshold(image, block_size=11, c_value=2,
                       method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C):
    """
    Apply adaptive thresholding.

    Computes a local threshold for each pixel based on its neighborhood,
    making it robust to uneven illumination and local intensity variations.

    Parameters
    ----------
    image : np.ndarray    Grayscale image (uint8).
    block_size : int      Size of the local neighborhood (must be odd, >= 3).
    c_value : int         Constant subtracted from the mean.
    method : int          cv2.ADAPTIVE_THRESH_GAUSSIAN_C or
                           cv2.ADAPTIVE_THRESH_MEAN_C.

    Returns
    -------
    np.ndarray  Binary mask (uint8).
    """
    # Ensure block_size is odd and >= 3
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    return cv2.adaptiveThreshold(
        image, 255, method, cv2.THRESH_BINARY, block_size, c_value
    )


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_iou(mask_pred, mask_gt):
    """
    Compute Intersection over Union (IoU) between predicted and ground-truth masks.

    IoU = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    mask_pred : np.ndarray  Predicted binary mask (0 or 255).
    mask_gt   : np.ndarray  Ground-truth binary mask (0 or 255).

    Returns
    -------
    float  IoU value in [0, 1]. Higher is better.
    """
    pred = (mask_pred > 127).astype(np.uint8)
    gt = (mask_gt > 127).astype(np.uint8)

    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)

    if union == 0:
        return 1.0  # Both masks are empty
    return intersection / union


def evaluate_segmentation(image, ground_truth=None, thresh_value=127,
                          block_size=11, c_value=2):
    """
    Apply all three thresholding methods and evaluate them.

    Parameters
    ----------
    image : np.ndarray
        Grayscale restored image.
    ground_truth : np.ndarray or None
        Ground-truth binary mask. If provided, IoU is computed.
    thresh_value : int
        Threshold for global thresholding.
    block_size : int
        Block size for adaptive thresholding.
    c_value : int
        C value for adaptive thresholding.

    Returns
    -------
    dict  {method_name: {'mask': binary_mask, 'iou': float or None, ...}}
    """
    results = {}

    # Global thresholding
    global_mask = global_threshold(image, thresh_value)
    results["Global"] = {
        "mask": global_mask,
        "params": f"threshold={thresh_value}",
    }

    # Otsu's thresholding
    otsu_mask, otsu_thresh = otsu_threshold(image)
    results["Otsu"] = {
        "mask": otsu_mask,
        "params": f"auto_threshold={otsu_thresh:.1f}",
    }

    # Adaptive thresholding
    adapt_mask = adaptive_threshold(image, block_size, c_value)
    results["Adaptive"] = {
        "mask": adapt_mask,
        "params": f"blockSize={block_size}, C={c_value}",
    }

    # Compute IoU if ground truth is available
    for name in results:
        if ground_truth is not None:
            results[name]["iou"] = compute_iou(results[name]["mask"], ground_truth)
        else:
            results[name]["iou"] = None

    return results


def select_best_segmentation(seg_results):
    """
    Select the best segmentation method.

    If IoU is available, picks highest IoU. Otherwise returns "Otsu" as
    a reasonable default (works well for bimodal distributions).

    Returns
    -------
    str  Name of the best method.
    """
    has_iou = any(r["iou"] is not None for r in seg_results.values())

    if has_iou:
        best = max(seg_results.items(), key=lambda x: x[1]["iou"] or 0)
        return best[0]
    else:
        # Default heuristic: Otsu is generally most reliable
        return "Otsu"
