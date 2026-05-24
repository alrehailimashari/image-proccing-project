"""
generate_images.py
==================
Loads existing images from images/raw/ if present,
otherwise generates synthetic grayscale images (512x512).

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

IMG_SIZE = 512
CENTER = IMG_SIZE // 2
BG_COLOR = 0
FG_COLOR = 255

RAW_DIR  = os.path.join("images", "raw")
MASK_DIR = os.path.join("images", "masks")


def _ensure_dirs():
    os.makedirs(RAW_DIR,  exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic generators (fallback only)
# ---------------------------------------------------------------------------
def generate_circle():
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)
    cv2.circle(img, (CENTER, CENTER), 150, FG_COLOR, thickness=-1)
    return img

def generate_wrench():
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)
    cv2.rectangle(img, (100, CENTER-20), (412, CENTER+20), FG_COLOR, -1)
    cv2.circle(img, (100, CENTER), 55, FG_COLOR, -1)
    cv2.circle(img, (412, CENTER), 45, FG_COLOR, -1)
    cv2.circle(img, (70,  CENTER), 25, BG_COLOR, -1)
    return img

def generate_hexagon():
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)
    radius = 160
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    pts = np.array([[CENTER + int(radius*np.cos(a)),
                     CENTER + int(radius*np.sin(a))] for a in angles], dtype=np.int32)
    cv2.fillPoly(img, [pts], FG_COLOR)
    return img

def generate_leaf():
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)
    cv2.ellipse(img, (CENTER, CENTER), (170, 90), 30, 0, 360, FG_COLOR, -1)
    tip = np.array([[CENTER+120,CENTER-100],[CENTER+170,CENTER-160],[CENTER+80,CENTER-130]], dtype=np.int32)
    cv2.fillPoly(img, [tip], FG_COLOR)
    stem = np.array([[CENTER-140,CENTER+70],[CENTER-190,CENTER+130],[CENTER-180,CENTER+140],[CENTER-130,CENTER+80]], dtype=np.int32)
    cv2.fillPoly(img, [stem], FG_COLOR)
    return img

def generate_washer():
    img = np.full((IMG_SIZE, IMG_SIZE), BG_COLOR, dtype=np.uint8)
    cv2.circle(img, (CENTER, CENTER), 160, FG_COLOR, -1)
    cv2.circle(img, (CENTER, CENTER),  60, BG_COLOR, -1)
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        bx = CENTER + int(110*np.cos(angle))
        by = CENTER + int(110*np.sin(angle))
        cv2.circle(img, (bx, by), 18, BG_COLOR, -1)
    return img


GENERATORS = {
    "circle":  generate_circle,
    "wrench":  generate_wrench,
    "hexagon": generate_hexagon,
    "leaf":    generate_leaf,
    "washer":  generate_washer,
}


# ---------------------------------------------------------------------------
# Main — use real image if exists, else generate synthetic
# ---------------------------------------------------------------------------
def generate_all():
    """
    For each object:
      - If images/raw/<name>.png already exists → use it (real image).
      - Otherwise → generate synthetic and save it.
    Always regenerate the mask from whatever image is used.
    """
    _ensure_dirs()
    paths = {}

    for name, gen_fn in GENERATORS.items():
        img_path  = os.path.join(RAW_DIR,  f"{name}.png")
        mask_path = os.path.join(MASK_DIR, f"{name}_mask.png")

        # ── Load real image if present ──────────────────────────────────
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [!] Could not read {img_path}, generating synthetic.")
                img = gen_fn()
                cv2.imwrite(img_path, img)
            else:
                # Resize to 512×512 if needed
                if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    cv2.imwrite(img_path, img)
                print(f"  [✓] {name:10s} <- real image  ({img_path})")
        else:
            # ── Generate synthetic ───────────────────────────────────────
            img = gen_fn()
            cv2.imwrite(img_path, img)
            print(f"  [+] {name:10s} -> synthetic  ({img_path})")

        # ── Generate mask from the image (Otsu threshold) ───────────────
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(mask_path, mask)

        paths[name] = {"image": img_path, "mask": mask_path}

    print(f"\nReady: {len(paths)} images + masks.")
    return paths


if __name__ == "__main__":
    generate_all()
