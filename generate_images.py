"""
generate_images.py
==================
Generates 5 clean synthetic grayscale images (512x512) and their
corresponding ground-truth binary masks.

Objects:
  1. Circle   — circular object
  2. Wrench   — elongated object
  3. Hexagon  — polygonal / angular object
  4. Leaf     — irregular / organic object
  5. Washer   — hollow / perforated object
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 512
CENTER = IMG_SIZE // 2
BG_COLOR = 0        # black background
FG_COLOR = 255      # white foreground

RAW_DIR = os.path.join("images", "raw")
MASK_DIR = os.path.join("images", "masks")


def _ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Circle
# ---------------------------------------------------------------------------
def generate_circle():
    """Generate a filled circle (radius 150 px) centered in the image."""
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)
    cv2.circle(img, (CENTER, CENTER), 150, FG_COLOR, thickness=-1)
    return img


# ---------------------------------------------------------------------------
# 2. Wrench (elongated tool shape)
# ---------------------------------------------------------------------------
def generate_wrench():
    """
    Generate an elongated wrench shape.
    Composed of a rectangular shaft with circular heads at each end,
    one larger (open-end jaw) and one smaller (box-end ring).
    """
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)

    # Shaft — long horizontal rectangle
    shaft_top_left = (100, CENTER - 20)
    shaft_bottom_right = (412, CENTER + 20)
    cv2.rectangle(img, shaft_top_left, shaft_bottom_right, FG_COLOR, -1)

    # Left head — larger circle (open-end jaw)
    cv2.circle(img, (100, CENTER), 55, FG_COLOR, -1)

    # Right head — larger circle (box-end)
    cv2.circle(img, (412, CENTER), 45, FG_COLOR, -1)

    # Cut-out in left head to simulate jaw opening
    cv2.circle(img, (70, CENTER), 25, BG_COLOR, -1)

    return img


# ---------------------------------------------------------------------------
# 3. Hexagon
# ---------------------------------------------------------------------------
def generate_hexagon():
    """Generate a regular hexagon centered in the image."""
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)
    radius = 160
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices
    pts = np.array([
        [CENTER + int(radius * np.cos(a)), CENTER + int(radius * np.sin(a))]
        for a in angles
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], FG_COLOR)
    return img


# ---------------------------------------------------------------------------
# 4. Leaf (irregular / organic)
# ---------------------------------------------------------------------------
def generate_leaf():
    """
    Generate an organic leaf shape using smooth curves.
    Uses an ellipse-based approach with a pointed tip and curved veins.
    """
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)

    # Main leaf body — tilted ellipse
    cv2.ellipse(img, (CENTER, CENTER), (170, 90), angle=30,
                startAngle=0, endAngle=360, color=FG_COLOR, thickness=-1)

    # Pointed tip — triangle at upper-right
    tip_pts = np.array([
        [CENTER + 120, CENTER - 100],
        [CENTER + 170, CENTER - 160],
        [CENTER + 80, CENTER - 130],
    ], dtype=np.int32)
    cv2.fillPoly(img, [tip_pts], FG_COLOR)

    # Stem — thin rectangle at lower-left
    stem_pts = np.array([
        [CENTER - 140, CENTER + 70],
        [CENTER - 190, CENTER + 130],
        [CENTER - 180, CENTER + 140],
        [CENTER - 130, CENTER + 80],
    ], dtype=np.int32)
    cv2.fillPoly(img, [stem_pts], FG_COLOR)

    return img


# ---------------------------------------------------------------------------
# 5. Washer (hollow / perforated)
# ---------------------------------------------------------------------------
def generate_washer():
    """
    Generate a washer (ring) with a center hole and 4 small bolt holes.
    """
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)

    # Outer ring
    cv2.circle(img, (CENTER, CENTER), 160, FG_COLOR, -1)

    # Center hole
    cv2.circle(img, (CENTER, CENTER), 60, BG_COLOR, -1)

    # Four small bolt holes evenly spaced
    bolt_radius = 18
    bolt_distance = 110
    for angle in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        bx = CENTER + int(bolt_distance * np.cos(angle))
        by = CENTER + int(bolt_distance * np.sin(angle))
        cv2.circle(img, (bx, by), bolt_radius, BG_COLOR, -1)

    return img


# ---------------------------------------------------------------------------
# Main — generate all images and masks
# ---------------------------------------------------------------------------
GENERATORS = {
    "circle":  generate_circle,
    "wrench":  generate_wrench,
    "hexagon": generate_hexagon,
    "leaf":    generate_leaf,
    "washer":  generate_washer,
}


def generate_all():
    """Generate all 5 images and their binary masks, save to disk."""
    _ensure_dirs()
    paths = {}
    for name, gen_fn in GENERATORS.items():
        img = gen_fn()
        img_path = os.path.join(RAW_DIR, f"{name}.png")
        cv2.imwrite(img_path, img)

        # Ground-truth mask — simple threshold of the clean image
        _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        mask_path = os.path.join(MASK_DIR, f"{name}_mask.png")
        cv2.imwrite(mask_path, mask)

        paths[name] = {"image": img_path, "mask": mask_path}
        print(f"  [+] {name:10s} -> {img_path}")

    print(f"\nGenerated {len(paths)} images + masks.")
    return paths


if __name__ == "__main__":
    print("Generating synthetic images...")
    generate_all()
