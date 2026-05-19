"""
pipeline/features.py
====================
Shape feature extraction from binary masks.

Extracted features:
  1. Area          — total pixel count inside contour
  2. Perimeter     — arc length of the contour boundary
  3. Circularity   — 4pi * Area / Perimeter^2  (1.0 = perfect circle)
  4. Aspect Ratio  — width / height of bounding rectangle
  5. Solidity      — Area / Convex Hull Area
  6. Extent        — Area / Bounding Rectangle Area
  7. Num Holes     — count of inner contours (holes)
"""

import cv2
import numpy as np
import csv
import os


def extract_features(binary_mask):
    """Extract 7 shape features from the largest contour in a binary mask."""
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    # Find largest external contour
    largest_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    cnt = contours[largest_idx]
    area = cv2.contourArea(cnt)
    if area == 0:
        return None

    perimeter = cv2.arcLength(cnt, closed=True)
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    rect_area = w * h
    extent = area / rect_area if rect_area > 0 else 0.0

    # Count holes: child contours of the largest contour
    num_holes = 0
    if hierarchy is not None:
        for i in range(len(contours)):
            if hierarchy[0][i][3] == largest_idx:
                num_holes += 1

    return {
        "area": round(area, 2),
        "perimeter": round(perimeter, 2),
        "circularity": round(circularity, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "solidity": round(solidity, 4),
        "extent": round(extent, 4),
        "num_holes": num_holes,
    }


def extract_features_all(masks_dict):
    """Extract features from multiple masks. Returns {name: feature_dict}."""
    results = {}
    for name, mask in masks_dict.items():
        features = extract_features(mask)
        if features:
            results[name] = features
    return results


def features_to_table(features_dict):
    """Convert features to (headers, rows) table format."""
    headers = ["Object", "Area", "Perimeter", "Circularity",
               "Aspect Ratio", "Solidity", "Extent", "Holes"]
    rows = []
    for name, feats in features_dict.items():
        rows.append([name.capitalize(), feats["area"], feats["perimeter"],
                     feats["circularity"], feats["aspect_ratio"],
                     feats["solidity"], feats["extent"], feats["num_holes"]])
    return headers, rows


def save_features_csv(features_dict, filepath):
    """Save feature table to CSV."""
    headers, rows = features_to_table(features_dict)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return filepath


def features_to_vector(features_dict):
    """Convert to (names_list, feature_matrix) for classification."""
    names = []
    vectors = []
    keys = ["area", "perimeter", "circularity",
            "aspect_ratio", "solidity", "extent", "num_holes"]
    for name, feats in features_dict.items():
        names.append(name)
        vectors.append([feats[k] for k in keys])
    return names, np.array(vectors, dtype=np.float64)
