"""
main.py
=======
Batch pipeline runner for Advanced Digital Image Processing project.

Runs the complete pipeline for all 5 images:
  1. Generate / load images
  2. Inject noise
  3. Restore & enhance (spatial + frequency domain)
  4. Compare (MSE, PSNR, visual)
  5. Segment (Global, Otsu, Adaptive)
  6. Morphological cleanup
  7. Extract features
  8. Classify (PCA + Minimum Distance)
  9. Generate report
"""

import os
import sys
import cv2
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_images import generate_all, GENERATORS
from pipeline.noise import apply_noise, DEFAULT_NOISE_CONFIG
from pipeline.restoration import apply_all_spatial_filters, histogram_equalization, clahe
from pipeline.frequency_filters import (apply_all_frequency_filters, apply_fft_filter,
                                         notch_reject, ideal_lowpass, butterworth_lowpass)
from pipeline.edge_detection import sobel_edges, laplacian_edges
from pipeline.comparator import (compute_mse, compute_psnr, compute_difference_map,
                                  generate_comparison_figure, evaluate_all_filters)
from pipeline.segmentation import evaluate_segmentation, select_best_segmentation
from pipeline.morphology import evaluate_morphology, apply_all_morphology, boundary_extraction
from pipeline.features import extract_features, extract_features_all, save_features_csv, features_to_table
from pipeline.classification import run_classification, CLASS_LABELS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

# Object descriptions for report
OBJECT_PROFILES = {
    "circle": {
        "name": "Circle",
        "profile": "Circular",
        "description": "A filled circle with a single smooth boundary and constant curvature. "
                       "Useful for testing circularity-based feature extraction and evaluating "
                       "how filters preserve curved edges.",
    },
    "wrench": {
        "name": "Wrench",
        "profile": "Elongated",
        "description": "An elongated tool shape with high aspect ratio and a mix of straight "
                       "and curved boundaries. Useful for testing aspect ratio features and "
                       "evaluating filter performance on thin structures.",
    },
    "hexagon": {
        "name": "Hexagon",
        "profile": "Polygonal",
        "description": "A regular six-sided polygon with straight edges meeting at 120-degree "
                       "angles. Useful for testing corner preservation during filtering and "
                       "evaluating periodic noise removal via FFT.",
    },
    "leaf": {
        "name": "Leaf",
        "profile": "Irregular",
        "description": "An organic leaf shape with asymmetric curves and no straight edges. "
                       "Useful for testing adaptive methods on complex, natural boundaries "
                       "and evaluating solidity/extent features.",
    },
    "washer": {
        "name": "Washer",
        "profile": "Hollow",
        "description": "A ring (donut) shape with a center hole and 4 bolt holes. Useful for "
                       "testing hole detection, evaluating filter preservation of internal "
                       "boundaries, and computing solidity features.",
    },
}


def ensure_dirs():
    """Create all output directories."""
    for d in [FIGURES_DIR, TABLES_DIR, os.path.join(OUTPUT_DIR, "pca")]:
        os.makedirs(d, exist_ok=True)


def run_pipeline():
    """Execute the full image processing pipeline."""
    ensure_dirs()

    # ===== STEP 1: Generate images =====
    print("=" * 60)
    print("STEP 1: Generating synthetic images")
    print("=" * 60)
    image_paths = generate_all()

    # Load all images and masks
    images = {}
    masks = {}
    for name in GENERATORS:
        images[name] = cv2.imread(image_paths[name]["image"], cv2.IMREAD_GRAYSCALE)
        masks[name] = cv2.imread(image_paths[name]["mask"], cv2.IMREAD_GRAYSCALE)

    # Storage for pipeline results
    pipeline_results = {}

    for obj_name in GENERATORS:
        print(f"\n{'=' * 60}")
        print(f"Processing: {obj_name.upper()}")
        print(f"{'=' * 60}")

        original = images[obj_name]
        gt_mask = masks[obj_name]
        noise_cfg = DEFAULT_NOISE_CONFIG[obj_name]
        info = OBJECT_PROFILES[obj_name]

        result = {"object_info": info, "noise_config": noise_cfg}

        # ===== STEP 2: Noise injection =====
        print(f"  [2] Injecting {noise_cfg['type']} noise...")
        noisy = apply_noise(original, noise_cfg["type"], **noise_cfg["params"])
        result["noisy"] = noisy

        # ===== STEP 3: Restoration =====
        print(f"  [3] Applying spatial and frequency filters...")

        # Spatial filters
        spatial_results = apply_all_spatial_filters(noisy)

        # Frequency filters
        freq_results = apply_all_frequency_filters(noisy)

        # Special: notch filter for periodic noise (hexagon)
        if noise_cfg["type"] == "periodic":
            freq_param = noise_cfg["params"].get("frequency", 30)
            for bw in [5, 10, 15]:
                nmask = notch_reject(noisy.shape, center_freq=freq_param, bandwidth=bw)
                freq_results[f"Notch BW={bw}"] = apply_fft_filter(noisy, nmask)

        # Combine all filter results
        all_filter_results = {**spatial_results, **freq_results}

        # Enhancement (apply to top spatial results)
        best_spatial_name = None
        best_spatial_psnr = -1
        for fname, fimg in spatial_results.items():
            p = compute_psnr(original, fimg)
            if p > best_spatial_psnr:
                best_spatial_psnr = p
                best_spatial_name = fname

        if best_spatial_name:
            enhanced_heq = histogram_equalization(spatial_results[best_spatial_name])
            enhanced_clahe = clahe(spatial_results[best_spatial_name])
            all_filter_results[f"{best_spatial_name} + HistEq"] = enhanced_heq
            all_filter_results[f"{best_spatial_name} + CLAHE"] = enhanced_clahe

        # ===== STEP 4: Comparison =====
        print(f"  [4] Evaluating filters (MSE/PSNR)...")
        eval_table = evaluate_all_filters(original, noisy, all_filter_results)
        result["eval_table"] = eval_table

        # Best filter (highest PSNR)
        best_entry = eval_table[0]
        best_filter_name = best_entry["filter"]
        best_restored = all_filter_results.get(best_filter_name, noisy)
        result["best_filter"] = best_filter_name
        result["best_restored"] = best_restored
        result["mse"] = best_entry["mse"]
        result["psnr"] = best_entry["psnr"]

        # Multi-criteria: also check segmentation quality of top 3 filters
        top3 = [e for e in eval_table[:4] if e["filter"] != "Noisy (no filter)"][:3]
        best_seg_iou = -1
        for entry in top3:
            fname = entry["filter"]
            fimg = all_filter_results.get(fname, noisy)
            seg = evaluate_segmentation(fimg, gt_mask)
            best_method = select_best_segmentation(seg)
            iou = seg[best_method]["iou"] or 0
            if iou > best_seg_iou:
                best_seg_iou = iou
                best_filter_name = fname
                best_restored = fimg

        result["best_filter"] = best_filter_name
        result["best_restored"] = best_restored
        result["mse"] = compute_mse(original, best_restored)
        result["psnr"] = compute_psnr(original, best_restored)

        print(f"    Best filter: {best_filter_name} "
              f"(PSNR={result['psnr']:.2f} dB, MSE={result['mse']:.2f})")

        # Difference map
        diff_map = compute_difference_map(original, best_restored)
        result["diff_map"] = diff_map

        # Save comparison figure
        fig_path = os.path.join(FIGURES_DIR, f"{obj_name}_comparison.png")
        generate_comparison_figure(
            original, noisy, best_restored, diff_map,
            result["mse"], result["psnr"],
            info["name"], noise_cfg["type"], best_filter_name,
            save_path=fig_path
        )
        result["comparison_fig"] = fig_path

        # Edge detection (for analysis, not restoration)
        edge_mag, _, _ = sobel_edges(best_restored)
        lap_edges = laplacian_edges(best_restored)

        # ===== STEP 5: Segmentation =====
        print(f"  [5] Segmenting...")
        seg_results = evaluate_segmentation(best_restored, gt_mask)
        best_seg_method = select_best_segmentation(seg_results)
        result["segmentation"] = seg_results
        result["best_seg_method"] = best_seg_method
        best_seg_mask = seg_results[best_seg_method]["mask"]
        result["best_seg_mask"] = best_seg_mask

        seg_iou = seg_results[best_seg_method]["iou"]
        print(f"    Best segmentation: {best_seg_method} (IoU={seg_iou:.4f})")

        # ===== STEP 6: Morphology =====
        print(f"  [6] Morphological processing...")
        morph_eval = evaluate_morphology(best_seg_mask, gt_mask)
        result["morphology"] = morph_eval
        best_morph_mask = morph_eval.get("best_mask", best_seg_mask)
        result["best_morph_mask"] = best_morph_mask

        print(f"    Best morphology: {morph_eval.get('best_operation', 'N/A')} "
              f"({morph_eval.get('best_kernel_shape', '')} "
              f"{morph_eval.get('best_kernel_size', '')}x"
              f"{morph_eval.get('best_kernel_size', '')}) "
              f"IoU={morph_eval.get('best_iou', 0):.4f}")

        # Boundary extraction
        boundary = boundary_extraction(best_morph_mask)
        result["boundary"] = boundary

        # Save segmentation/morphology figure
        fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
        fig2.suptitle(f"{info['name']} — Segmentation & Morphology", fontsize=13, fontweight='bold')
        panels = [
            (best_restored, f"Restored"),
            (best_seg_mask, f"Seg: {best_seg_method}"),
            (best_morph_mask, f"Morph: {morph_eval.get('best_operation', 'N/A')}"),
            (boundary, "Boundary"),
        ]
        for ax, (img, title) in zip(axes2, panels):
            ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        plt.tight_layout()
        seg_fig_path = os.path.join(FIGURES_DIR, f"{obj_name}_seg_morph.png")
        fig2.savefig(seg_fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        result["seg_morph_fig"] = seg_fig_path

        # ===== STEP 7: Feature Extraction =====
        print(f"  [7] Extracting features...")
        features = extract_features(best_morph_mask)
        result["features"] = features
        if features:
            print(f"    Area={features['area']}, Circularity={features['circularity']:.4f}, "
                  f"Holes={features['num_holes']}")

        pipeline_results[obj_name] = result

    # ===== STEP 8: Classification =====
    print(f"\n{'=' * 60}")
    print("STEP 8: Classification")
    print("=" * 60)

    all_features = {name: r["features"] for name, r in pipeline_results.items()
                    if r["features"] is not None}
    save_features_csv(all_features, os.path.join(TABLES_DIR, "features.csv"))

    clf_output = run_classification(all_features, save_dir=OUTPUT_DIR)
    print(clf_output["report"])

    # ===== STEP 9: Generate Report =====
    print(f"\n{'=' * 60}")
    print("STEP 9: Generating report")
    print("=" * 60)
    generate_report(pipeline_results, all_features, clf_output)

    print("\n[DONE] Pipeline complete. Check output/ and report.md")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _param_justification(obj_name, result):
    """Generate parameter justification text for a given object."""
    noise_type = result["noise_config"]["type"]
    best_filter = result["best_filter"]
    best_seg = result["best_seg_method"]
    morph = result["morphology"]

    lines = []

    # Filter justification
    if "Median" in best_filter:
        lines.append(f"- **{best_filter}** was selected because median filtering is the most effective "
                     f"method for impulse-type noise, as it replaces outlier pixels with the "
                     f"neighborhood median without blurring edges.")
        if "5" in best_filter:
            lines.append(f"  - A 5×5 kernel outperformed 3×3 because the noise density required a "
                         f"larger neighborhood to reliably find a clean median value.")
    elif "Gaussian" in best_filter:
        lines.append(f"- **{best_filter}** was selected because Gaussian blur effectively "
                     f"averages out additive Gaussian noise with a weighted kernel that "
                     f"preserves more central detail than a mean filter.")
    elif "Notch" in best_filter:
        lines.append(f"- **{best_filter}** was selected because the periodic noise creates "
                     f"distinct peaks in the frequency spectrum that can be precisely "
                     f"targeted and removed by a notch filter, leaving other frequencies intact.")
    elif "Butterworth" in best_filter or "LP" in best_filter:
        lines.append(f"- **{best_filter}** was selected because frequency-domain low-pass "
                     f"filtering smooths the noise while the Butterworth roll-off avoids "
                     f"ringing artifacts that ideal filters produce.")
    else:
        lines.append(f"- **{best_filter}** provided the best combination of noise reduction "
                     f"and boundary preservation for {noise_type} noise.")

    # Segmentation justification
    if best_seg == "Otsu":
        lines.append(f"- **Otsu thresholding** was optimal because the restored image has a "
                     f"clear bimodal histogram (object vs background), allowing Otsu's "
                     f"algorithm to find the ideal threshold automatically.")
    elif best_seg == "Global":
        lines.append(f"- **Global thresholding** performed best due to uniform illumination "
                     f"and clear intensity separation between object and background.")
    elif best_seg == "Adaptive":
        lines.append(f"- **Adaptive thresholding** was needed because local intensity "
                     f"variations after filtering made a single global threshold insufficient.")

    # Morphology justification
    best_op = morph.get("best_operation", "N/A")
    best_ks = morph.get("best_kernel_size", "N/A")
    best_kshape = morph.get("best_kernel_shape", "N/A")
    if best_op == "Opening":
        lines.append(f"- **Opening ({best_kshape} {best_ks}×{best_ks})** was chosen because "
                     f"it removes small noise blobs (via erosion) then restores object size "
                     f"(via dilation), effectively cleaning the mask without shrinking it.")
    elif best_op == "Closing":
        lines.append(f"- **Closing ({best_kshape} {best_ks}×{best_ks})** was chosen because "
                     f"it fills small gaps and holes in the segmentation mask, producing a "
                     f"more complete object representation.")
    else:
        lines.append(f"- **{best_op} ({best_kshape} {best_ks}×{best_ks})** produced the "
                     f"cleanest mask with the highest IoU against ground truth.")

    return "\n".join(lines)


def generate_report(pipeline_results, all_features, clf_output):
    """Generate the academic-style report as report.md."""
    lines = []

    # Title
    lines.append("# Advanced Digital Image Processing — Project Report\n")
    lines.append("## 1. Introduction\n")
    lines.append("This report presents a complete digital image processing pipeline applied to "
                 "5 synthetic object images. The pipeline covers noise injection, spatial and "
                 "frequency-domain restoration, segmentation, morphological processing, feature "
                 "extraction, and classification using PCA and Minimum Distance methods.\n")

    # Dataset description
    lines.append("## 2. Dataset Description\n")
    lines.append("| # | Object | Geometric Profile | Description |")
    lines.append("|---|--------|-------------------|-------------|")
    for i, (name, info) in enumerate(OBJECT_PROFILES.items(), 1):
        lines.append(f"| {i} | {info['name']} | {info['profile']} | {info['description'][:80]}... |")
    lines.append("")

    # Per-image analysis
    lines.append("## 3. Per-Image Analysis\n")

    for obj_name, result in pipeline_results.items():
        info = result["object_info"]
        noise_cfg = result["noise_config"]
        noise_params = ", ".join(f"{k}={v}" for k, v in noise_cfg["params"].items())

        lines.append(f"### 3.{list(GENERATORS).index(obj_name)+1}. {info['name']} ({info['profile']})\n")
        lines.append(f"**Noise type:** {noise_cfg['type']} ({noise_params})\n")
        lines.append(f"**Best filter:** {result['best_filter']}\n")
        lines.append(f"**MSE:** {result['mse']:.2f} | **PSNR:** {result['psnr']:.2f} dB\n")
        lines.append(f"**Best segmentation:** {result['best_seg_method']} "
                     f"(IoU = {result['segmentation'][result['best_seg_method']]['iou']:.4f})\n")

        morph = result["morphology"]
        lines.append(f"**Best morphology:** {morph.get('best_operation', 'N/A')} "
                     f"({morph.get('best_kernel_shape', '')} "
                     f"{morph.get('best_kernel_size', '')}×{morph.get('best_kernel_size', '')}) "
                     f"IoU = {morph.get('best_iou', 0):.4f}\n")

        # Comparison figure
        lines.append(f"![{info['name']} Comparison]({result['comparison_fig']})\n")
        lines.append(f"![{info['name']} Segmentation & Morphology]({result['seg_morph_fig']})\n")

        # Parameter justification
        lines.append("#### Parameter Justification\n")
        lines.append(_param_justification(obj_name, result))
        lines.append("")

    # Group Summary Table
    lines.append("## 4. Group Summary Table\n")
    lines.append("| Image | Object | Profile | Noise | Best Filter | Best Threshold | "
                 "Best Morphology | PSNR (dB) | MSE | Predicted Class |")
    lines.append("|-------|--------|---------|-------|-------------|----------------|"
                 "-----------------|-----------|-----|-----------------|")

    clf_map = {}
    if clf_output and clf_output.get("classification_results"):
        for cr in clf_output["classification_results"]:
            clf_map[cr["name"]] = cr["predicted_class"]

    for obj_name, result in pipeline_results.items():
        info = result["object_info"]
        morph = result["morphology"]
        pred_class = clf_map.get(obj_name, "N/A")
        morph_str = f"{morph.get('best_operation', 'N/A')} {morph.get('best_kernel_shape', '')} {morph.get('best_kernel_size', '')}x{morph.get('best_kernel_size', '')}"
        lines.append(f"| {info['name']} | {info['name']} | {info['profile']} | "
                     f"{result['noise_config']['type']} | {result['best_filter']} | "
                     f"{result['best_seg_method']} | {morph_str} | "
                     f"{result['psnr']:.2f} | {result['mse']:.2f} | {pred_class} |")
    lines.append("")

    # Feature table
    lines.append("## 5. Feature Extraction\n")
    headers, rows = features_to_table(all_features)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    lines.append("")

    # PCA
    lines.append("## 6. PCA Analysis\n")
    lines.append(f"![PCA Scatter Plot]({clf_output.get('pca_path', 'output/pca/pca_scatter.png')})\n")
    lines.append("The PCA scatter plot projects the 7-dimensional feature space onto "
                 "2 principal components. Points are colored by geometric class. "
                 "Clear separation between classes indicates that the extracted shape "
                 "features effectively distinguish different object types.\n")

    # Classification
    lines.append("## 7. Classification Report\n")
    lines.append("```")
    lines.append(clf_output.get("report", "No classification results."))
    lines.append("```\n")
    lines.append(f"**Overall accuracy:** {clf_output.get('accuracy', 0)*100:.1f}%\n")

    # Conclusion
    lines.append("## 8. Conclusion\n")
    lines.append("This project demonstrated a complete image processing pipeline from noise "
                 "injection through classification. Key findings:\n")
    lines.append("- Median filtering is most effective for salt-and-pepper noise.")
    lines.append("- Notch filtering in the frequency domain precisely removes periodic noise.")
    lines.append("- Otsu thresholding provides robust automatic segmentation for bimodal images.")
    lines.append("- Morphological opening/closing effectively cleans segmentation masks.")
    lines.append("- Shape features (circularity, aspect ratio, solidity) provide strong "
                 "discriminative power for geometric classification.\n")

    # Write report
    report_text = "\n".join(lines)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("  [+] Report saved to report.md")


if __name__ == "__main__":
    run_pipeline()
