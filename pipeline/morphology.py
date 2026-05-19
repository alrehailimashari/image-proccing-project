"""
pipeline/morphology.py
======================
Morphological processing for binary mask cleanup.

Operations:
  - Erosion   — removes small foreground blobs / noise
  - Dilation  — fills small holes / gaps in foreground
  - Opening   — erosion then dilation (noise removal)
  - Closing   — dilation then erosion (gap filling)
  - Boundary Extraction — original minus eroded

All operations support adjustable kernel shapes and sizes.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Kernel construction
# ---------------------------------------------------------------------------

KERNEL_SHAPES = {
    "rect":    cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross":   cv2.MORPH_CROSS,
}


def get_kernel(shape="rect", size=3):
    """
    Create a structuring element (kernel) for morphological operations.

    Parameters
    ----------
    shape : str   One of 'rect', 'ellipse', 'cross'.
    size  : int   Kernel size (creates a size × size kernel).

    Returns
    -------
    np.ndarray  Structuring element.
    """
    if shape not in KERNEL_SHAPES:
        raise ValueError(f"Unknown kernel shape: {shape}. "
                         f"Choose from {list(KERNEL_SHAPES.keys())}")
    return cv2.getStructuringElement(KERNEL_SHAPES[shape], (size, size))


# ---------------------------------------------------------------------------
# Morphological operations
# ---------------------------------------------------------------------------

def erosion(image, kernel_shape="rect", kernel_size=3, iterations=1):
    """
    Apply erosion to a binary mask.

    Erosion shrinks foreground objects. A pixel is kept as foreground only
    if all pixels under the kernel are foreground. Removes small noise blobs.

    Parameters
    ----------
    image : np.ndarray        Binary mask (uint8, 0 or 255).
    kernel_shape : str        Structuring element shape.
    kernel_size : int         Structuring element size.
    iterations : int          Number of erosion iterations.

    Returns
    -------
    np.ndarray  Eroded mask.
    """
    kernel = get_kernel(kernel_shape, kernel_size)
    return cv2.erode(image, kernel, iterations=iterations)


def dilation(image, kernel_shape="rect", kernel_size=3, iterations=1):
    """
    Apply dilation to a binary mask.

    Dilation expands foreground objects. A pixel becomes foreground if any
    pixel under the kernel is foreground. Fills small holes and gaps.

    Parameters
    ----------
    image : np.ndarray        Binary mask.
    kernel_shape : str        Structuring element shape.
    kernel_size : int         Structuring element size.
    iterations : int          Number of dilation iterations.

    Returns
    -------
    np.ndarray  Dilated mask.
    """
    kernel = get_kernel(kernel_shape, kernel_size)
    return cv2.dilate(image, kernel, iterations=iterations)


def opening(image, kernel_shape="rect", kernel_size=3):
    """
    Apply morphological opening (erosion followed by dilation).

    Removes small foreground noise blobs while approximately preserving
    the size and shape of larger objects.

    Parameters
    ----------
    image : np.ndarray        Binary mask.
    kernel_shape : str        Structuring element shape.
    kernel_size : int         Structuring element size.

    Returns
    -------
    np.ndarray  Opened mask.
    """
    kernel = get_kernel(kernel_shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def closing(image, kernel_shape="rect", kernel_size=3):
    """
    Apply morphological closing (dilation followed by erosion).

    Fills small holes and gaps in the foreground while approximately
    preserving the size and shape of the object.

    Parameters
    ----------
    image : np.ndarray        Binary mask.
    kernel_shape : str        Structuring element shape.
    kernel_size : int         Structuring element size.

    Returns
    -------
    np.ndarray  Closed mask.
    """
    kernel = get_kernel(kernel_shape, kernel_size)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def boundary_extraction(image, kernel_shape="rect", kernel_size=3):
    """
    Extract object boundary using morphological operations.

    Boundary = Original − Eroded.
    This gives a thin outline of the object.

    Parameters
    ----------
    image : np.ndarray        Binary mask.
    kernel_shape : str        Structuring element shape.
    kernel_size : int         Structuring element size.

    Returns
    -------
    np.ndarray  Boundary image (uint8).
    """
    eroded = erosion(image, kernel_shape, kernel_size)
    return cv2.subtract(image, eroded)


# ---------------------------------------------------------------------------
# Apply all operations and evaluate
# ---------------------------------------------------------------------------

MORPH_OPERATIONS = {
    "Erosion":    erosion,
    "Dilation":   dilation,
    "Opening":    opening,
    "Closing":    closing,
    "Boundary":   boundary_extraction,
}


def apply_all_morphology(binary_mask, kernel_shape="rect", kernel_size=3):
    """
    Apply all morphological operations with the given kernel.

    Returns
    -------
    dict  {operation_name: result_mask}
    """
    results = {}
    for name, func in MORPH_OPERATIONS.items():
        results[name] = func(binary_mask, kernel_shape, kernel_size)
    return results


def evaluate_morphology(binary_mask, ground_truth, kernel_shapes=None, kernel_sizes=None):
    """
    Try all combinations of operations, kernel shapes, and sizes.
    Select the best by IoU against ground truth.

    Returns
    -------
    dict with keys: 'best_operation', 'best_kernel_shape', 'best_kernel_size',
                    'best_iou', 'best_mask', 'all_results'
    """
    from pipeline.segmentation import compute_iou

    if kernel_shapes is None:
        kernel_shapes = ["rect", "ellipse", "cross"]
    if kernel_sizes is None:
        kernel_sizes = [3, 5, 7]

    best_iou = -1.0
    best_info = {}
    all_results = []

    # Exclude boundary from "cleaning" evaluation (it's for visualization)
    cleaning_ops = {k: v for k, v in MORPH_OPERATIONS.items() if k != "Boundary"}

    for op_name, op_func in cleaning_ops.items():
        for shape in kernel_shapes:
            for size in kernel_sizes:
                result = op_func(binary_mask, shape, size)
                iou = compute_iou(result, ground_truth)

                entry = {
                    "operation": op_name,
                    "kernel_shape": shape,
                    "kernel_size": size,
                    "iou": iou,
                }
                all_results.append(entry)

                if iou > best_iou:
                    best_iou = iou
                    best_info = {
                        "best_operation": op_name,
                        "best_kernel_shape": shape,
                        "best_kernel_size": size,
                        "best_iou": iou,
                        "best_mask": result,
                    }

    best_info["all_results"] = all_results
    return best_info
