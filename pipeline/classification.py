"""
pipeline/classification.py
===========================
PCA visualization and Minimum Distance Classification.

Geometric profile classes:
  - Circular   (circle)
  - Elongated  (wrench)
  - Polygonal  (hexagon)
  - Irregular  (leaf)
  - Hollow     (washer)

Workflow:
  1. Standardize features with StandardScaler
  2. Generate augmented training data for robust centroids
  3. Compute class centroids from augmented training set
  4. Classify each original sample by nearest centroid
  5. PCA for 2D visualization
  6. Classification report
"""

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Ground-truth class mapping
CLASS_LABELS = {
    "circle":  "Circular",
    "wrench":  "Elongated",
    "hexagon": "Polygonal",
    "leaf":    "Irregular",
    "washer":  "Hollow",
}

CLASS_COLORS = {
    "Circular":  "#e74c3c",
    "Elongated": "#3498db",
    "Polygonal": "#2ecc71",
    "Irregular": "#f39c12",
    "Hollow":    "#9b59b6",
}


def standardize_features(feature_matrix):
    """Standardize features to zero mean and unit variance."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    return scaled, scaler


def augment_features(feature_matrix, names, true_classes, n_augments=10,
                     noise_std=0.05, seed=42):
    """
    Generate augmented training samples by adding small perturbations.

    For each original sample, create n_augments copies with Gaussian noise
    added to simulate natural variation (e.g. from different noise levels,
    filter parameters, or image orientations).
    """
    rng = np.random.RandomState(seed)
    aug_features = []
    aug_names = []
    aug_classes = []

    for i in range(len(names)):
        for j in range(n_augments):
            noise = rng.normal(0, noise_std, feature_matrix.shape[1])
            perturbed = feature_matrix[i] + noise * np.abs(feature_matrix[i] + 1e-8)
            aug_features.append(perturbed)
            aug_names.append(f"{names[i]}_aug{j}")
            aug_classes.append(true_classes[i])

    return np.array(aug_features), aug_names, aug_classes


def compute_pca(feature_matrix, n_components=2):
    """Apply PCA to reduce feature space to n_components dimensions."""
    pca = PCA(n_components=n_components)
    result = pca.fit_transform(feature_matrix)
    return result, pca


def plot_pca(pca_coords, names, true_classes, pca_obj, save_path=None):
    """Plot PCA scatter with points colored by geometric class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("PCA - Feature Space Visualization", fontsize=14, fontweight='bold')

    for i, (x, y) in enumerate(pca_coords):
        cls = true_classes[i]
        color = CLASS_COLORS.get(cls, "#888888")
        ax.scatter(x, y, c=color, s=200, edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(names[i].capitalize(), (x, y),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=10, fontweight='bold')

    for cls, color in CLASS_COLORS.items():
        ax.scatter([], [], c=color, s=100, edgecolors='black', label=cls)
    ax.legend(title="Geometric Class", loc="best", fontsize=9)

    var1 = pca_obj.explained_variance_ratio_[0] * 100
    var2 = pca_obj.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def minimum_distance_classify(test_features, test_names, test_classes,
                               train_features, train_classes):
    """
    Minimum Distance Classification.

    Computes class centroids from the training set, then classifies each
    test sample by finding the nearest centroid (Euclidean distance).
    """
    unique_classes = list(set(train_classes))

    # Compute class centroids from training data
    centroids = {}
    for cls in unique_classes:
        cls_features = np.array([train_features[j]
                                 for j in range(len(train_classes))
                                 if train_classes[j] == cls])
        centroids[cls] = np.mean(cls_features, axis=0)

    # Classify each test sample
    results = []
    for i in range(len(test_names)):
        test_vec = test_features[i]

        distances = {}
        for cls, centroid in centroids.items():
            distances[cls] = float(np.linalg.norm(test_vec - centroid))

        predicted = min(distances, key=distances.get)

        results.append({
            "name": test_names[i],
            "true_class": test_classes[i],
            "predicted_class": predicted,
            "correct": predicted == test_classes[i],
            "distances": {k: round(v, 4) for k, v in distances.items()},
            "min_distance": round(distances[predicted], 4),
        })

    return results


def generate_classification_report(results, save_path=None):
    """Generate a formatted classification report."""
    lines = []
    lines.append("=" * 70)
    lines.append("CLASSIFICATION REPORT - Minimum Distance Classifier")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Method: Class centroids computed from augmented training data.")
    lines.append("Distance metric: Euclidean distance in standardized feature space.")
    lines.append("")

    correct = sum(1 for r in results if r["correct"])
    total = len(results)

    for r in results:
        status = "CORRECT" if r["correct"] else "WRONG"
        lines.append(f"Object: {r['name'].capitalize()}")
        lines.append(f"  True class:      {r['true_class']}")
        lines.append(f"  Predicted class:  {r['predicted_class']}  [{status}]")
        lines.append(f"  Min distance:     {r['min_distance']}")
        dist_str = ", ".join(f"{k}: {v}" for k, v in r['distances'].items())
        lines.append(f"  All distances:    {dist_str}")
        lines.append("")

    lines.append(f"Overall Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

    return report_text


def run_classification(features_dict, save_dir="output"):
    """
    Full classification pipeline:
      1. Extract feature vectors
      2. Generate augmented training data
      3. Standardize all features together
      4. Compute PCA on original samples
      5. Classify originals using augmented training centroids
      6. Generate report
    """
    from pipeline.features import features_to_vector

    names, feature_matrix = features_to_vector(features_dict)
    true_classes = [CLASS_LABELS.get(n, "Unknown") for n in names]

    # 1. Augment training data (10 perturbed copies per sample)
    aug_features, aug_names, aug_classes = augment_features(
        feature_matrix, names, true_classes, n_augments=10, noise_std=0.05
    )

    # 2. Combine original + augmented for standardization
    all_features = np.vstack([feature_matrix, aug_features])
    n_orig = len(names)

    # 3. Standardize using all data
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_features)
    test_scaled = all_scaled[:n_orig]
    train_scaled = all_scaled[n_orig:]

    # 4. PCA on original samples only
    n_comp = min(2, test_scaled.shape[1], test_scaled.shape[0])
    pca_coords, pca_obj = compute_pca(test_scaled, n_components=n_comp)
    pca_path = os.path.join(save_dir, "pca", "pca_scatter.png")
    plot_pca(pca_coords, names, true_classes, pca_obj, save_path=pca_path)

    # 5. Classify
    clf_results = minimum_distance_classify(
        test_scaled, names, true_classes,
        train_scaled, aug_classes
    )

    # 6. Report
    report_path = os.path.join(save_dir, "tables", "classification_report.txt")
    report_text = generate_classification_report(clf_results, save_path=report_path)

    accuracy = sum(1 for r in clf_results if r["correct"]) / len(clf_results)

    return {
        "pca_coords": pca_coords,
        "classification_results": clf_results,
        "report": report_text,
        "accuracy": accuracy,
        "pca_path": pca_path,
    }
