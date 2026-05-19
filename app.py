"""
app.py
======
Gradio interactive interface for the image processing pipeline.

Provides real-time control over all processing parameters:
  - Noise type & intensity
  - Spatial filter type & kernel size
  - Frequency filter type & cutoff D0
  - Notch filter parameters
  - Thresholding method & parameters
  - Morphology operation, kernel shape & size

Displays: Original, Noisy, Restored, Difference Map,
          Segmentation Mask, Morphology Result, MSE, PSNR, Features.
"""

import os
import sys
import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_images import generate_all, GENERATORS
from pipeline.noise import apply_noise
from pipeline.restoration import mean_filter, gaussian_filter, median_filter
from pipeline.frequency_filters import (apply_fft_filter, ideal_lowpass,
                                         butterworth_lowpass, ideal_highpass,
                                         notch_reject)
from pipeline.comparator import compute_mse, compute_psnr, compute_difference_map
from pipeline.segmentation import global_threshold, otsu_threshold, adaptive_threshold
from pipeline.morphology import (erosion, dilation, opening, closing,
                                  boundary_extraction)
from pipeline.features import extract_features

# ---------------------------------------------------------------------------
# Generate images on startup
# ---------------------------------------------------------------------------
print("Loading images...")
image_paths = generate_all()
IMAGES = {}
for name in GENERATORS:
    IMAGES[name] = cv2.imread(image_paths[name]["image"], cv2.IMREAD_GRAYSCALE)
print("Images loaded.\n")

IMAGE_NAMES = list(IMAGES.keys())


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------
def process_image(
    image_name,
    # Noise
    noise_type, noise_density, noise_sigma, noise_freq, noise_amp,
    noise_gamma_shape, noise_gamma_scale, noise_uniform_low, noise_uniform_high,
    # Spatial filter
    spatial_filter, spatial_ksize, gaussian_sigma,
    # Frequency filter
    freq_filter, fft_d0, notch_center_freq, notch_bandwidth,
    # Enhancement
    apply_hist_eq, apply_clahe_flag,
    # Segmentation
    seg_method, global_thresh_val, adaptive_block_size, adaptive_c,
    # Morphology
    morph_op, morph_kernel_shape, morph_kernel_size,
):
    """Process a single image through the entire pipeline with given parameters."""

    original = IMAGES[image_name].copy()

    # --- 1. Noise Injection ---
    noise_params = {}
    if noise_type == "Salt & Pepper":
        noise_params = {"density": noise_density}
        noise_key = "salt_pepper"
    elif noise_type == "Gaussian":
        noise_params = {"mean": 0, "sigma": noise_sigma}
        noise_key = "gaussian"
    elif noise_type == "Periodic":
        noise_params = {"frequency": noise_freq, "amplitude": noise_amp}
        noise_key = "periodic"
    elif noise_type == "Gamma/Erlang":
        noise_params = {"shape": noise_gamma_shape, "scale": noise_gamma_scale}
        noise_key = "gamma"
    else:  # Uniform
        noise_params = {"low": noise_uniform_low, "high": noise_uniform_high}
        noise_key = "uniform"

    noisy = apply_noise(original, noise_key, **noise_params)

    # --- 2. Spatial Filter ---
    if spatial_filter == "Mean":
        restored = mean_filter(noisy, ksize=spatial_ksize)
    elif spatial_filter == "Gaussian":
        restored = gaussian_filter(noisy, ksize=spatial_ksize, sigma=gaussian_sigma)
    elif spatial_filter == "Median":
        restored = median_filter(noisy, ksize=spatial_ksize)
    else:  # None
        restored = noisy.copy()

    # --- 3. Frequency Filter ---
    if freq_filter != "None":
        shape = restored.shape
        if freq_filter == "Ideal Low-pass":
            mask = ideal_lowpass(shape, d0=fft_d0)
        elif freq_filter == "Butterworth Low-pass":
            mask = butterworth_lowpass(shape, d0=fft_d0, order=2)
        elif freq_filter == "Ideal High-pass":
            mask = ideal_highpass(shape, d0=fft_d0)
        elif freq_filter == "Notch Reject":
            mask = notch_reject(shape, center_freq=int(notch_center_freq),
                               bandwidth=int(notch_bandwidth))
        else:
            mask = np.ones(shape)
        restored = apply_fft_filter(restored, mask)

    # --- 4. Enhancement ---
    if apply_hist_eq:
        restored = cv2.equalizeHist(restored)
    if apply_clahe_flag:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        restored = clahe_obj.apply(restored)

    # --- 5. Metrics ---
    mse_val = compute_mse(original, restored)
    psnr_val = compute_psnr(original, restored)
    diff_map = compute_difference_map(original, restored)

    # --- 6. Segmentation ---
    if seg_method == "Global":
        seg_mask = global_threshold(restored, int(global_thresh_val))
    elif seg_method == "Otsu":
        seg_mask, _ = otsu_threshold(restored)
    else:  # Adaptive
        bs = int(adaptive_block_size)
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1
        seg_mask = adaptive_threshold(restored, block_size=bs, c_value=int(adaptive_c))

    # --- 7. Morphology ---
    if morph_op == "Erosion":
        morph_result = erosion(seg_mask, morph_kernel_shape, morph_kernel_size)
    elif morph_op == "Dilation":
        morph_result = dilation(seg_mask, morph_kernel_shape, morph_kernel_size)
    elif morph_op == "Opening":
        morph_result = opening(seg_mask, morph_kernel_shape, morph_kernel_size)
    elif morph_op == "Closing":
        morph_result = closing(seg_mask, morph_kernel_shape, morph_kernel_size)
    elif morph_op == "Boundary":
        morph_result = boundary_extraction(seg_mask, morph_kernel_shape, morph_kernel_size)
    else:
        morph_result = seg_mask.copy()

    # --- 8. Feature Extraction ---
    features = extract_features(morph_result)
    if features:
        feature_data = [[k.replace("_", " ").title(), str(v)]
                        for k, v in features.items()]
    else:
        feature_data = [["No contour found", "N/A"]]

    # Return all outputs
    return (
        original,       # Original image
        noisy,          # Noisy image
        restored,       # Restored image
        diff_map,       # Difference map
        seg_mask,       # Segmentation mask
        morph_result,   # Morphology result
        f"{mse_val:.2f}",    # MSE
        f"{psnr_val:.2f}",   # PSNR
        feature_data,   # Feature table
    )


# ---------------------------------------------------------------------------
# Build Gradio Interface
# ---------------------------------------------------------------------------
def build_app():
    """Build and return the Gradio Blocks app."""

    with gr.Blocks(
        title="Advanced Digital Image Processing Pipeline",
    ) as app:

        gr.Markdown(
            "# 🔬 Advanced Digital Image Processing Pipeline\n"
            "Interactive control over noise, filtering, segmentation, "
            "morphology, and feature extraction."
        )

        with gr.Row():
            # ============ LEFT COLUMN: Controls ============
            with gr.Column(scale=1):

                # Image Selection
                image_dd = gr.Dropdown(
                    choices=IMAGE_NAMES,
                    value=IMAGE_NAMES[0],
                    label="Select Image",
                )

                # --- Noise Controls ---
                with gr.Accordion("🔊 Noise Controls", open=True):
                    noise_type = gr.Dropdown(
                        choices=["Salt & Pepper", "Gaussian", "Periodic",
                                 "Gamma/Erlang", "Uniform"],
                        value="Salt & Pepper",
                        label="Noise Type",
                    )
                    noise_density = gr.Slider(0.01, 0.3, 0.05, step=0.01,
                                              label="S&P Density")
                    noise_sigma = gr.Slider(5, 80, 25, step=1,
                                            label="Gaussian Sigma")
                    noise_freq = gr.Slider(5, 100, 30, step=1,
                                           label="Periodic Frequency")
                    noise_amp = gr.Slider(5, 100, 40, step=1,
                                          label="Periodic Amplitude")
                    noise_gshape = gr.Slider(1, 20, 5, step=0.5,
                                             label="Gamma Shape")
                    noise_gscale = gr.Slider(1, 20, 5, step=0.5,
                                              label="Gamma Scale")
                    noise_ulow = gr.Slider(-100, 0, -30, step=1,
                                            label="Uniform Low")
                    noise_uhigh = gr.Slider(0, 100, 30, step=1,
                                             label="Uniform High")

                # --- Spatial Filter ---
                with gr.Accordion("🧹 Spatial Filter", open=True):
                    spatial_filter = gr.Dropdown(
                        choices=["None", "Mean", "Gaussian", "Median"],
                        value="Median",
                        label="Filter Type",
                    )
                    spatial_ksize = gr.Slider(3, 15, 5, step=2,
                                              label="Kernel Size")
                    gauss_sigma = gr.Slider(0.5, 5.0, 1.0, step=0.1,
                                            label="Gaussian Filter Sigma")

                # --- Frequency Filter ---
                with gr.Accordion("📡 Frequency Filter", open=True):
                    freq_filter = gr.Dropdown(
                        choices=["None", "Ideal Low-pass", "Butterworth Low-pass",
                                 "Ideal High-pass", "Notch Reject"],
                        value="None",
                        label="FFT Filter Type",
                    )
                    fft_d0 = gr.Slider(5, 200, 50, step=1,
                                       label="Cutoff D0")
                    notch_cf = gr.Slider(5, 200, 30, step=1,
                                         label="Notch Center Frequency")
                    notch_bw = gr.Slider(2, 50, 10, step=1,
                                          label="Notch Bandwidth")

                # --- Enhancement ---
                with gr.Accordion("✨ Enhancement", open=False):
                    hist_eq = gr.Checkbox(label="Histogram Equalization", value=False)
                    clahe_flag = gr.Checkbox(label="CLAHE", value=False)

                # --- Segmentation ---
                with gr.Accordion("✂️ Segmentation", open=True):
                    seg_method = gr.Dropdown(
                        choices=["Global", "Otsu", "Adaptive"],
                        value="Otsu",
                        label="Method",
                    )
                    global_thresh = gr.Slider(0, 255, 127, step=1,
                                              label="Global Threshold")
                    adapt_block = gr.Slider(3, 51, 11, step=2,
                                            label="Adaptive Block Size")
                    adapt_c = gr.Slider(-10, 20, 2, step=1,
                                        label="Adaptive C Value")

                # --- Morphology ---
                with gr.Accordion("🔧 Morphology", open=True):
                    morph_op = gr.Dropdown(
                        choices=["None", "Erosion", "Dilation", "Opening",
                                 "Closing", "Boundary"],
                        value="Opening",
                        label="Operation",
                    )
                    morph_kshape = gr.Dropdown(
                        choices=["rect", "ellipse", "cross"],
                        value="rect",
                        label="Kernel Shape",
                    )
                    morph_ksize = gr.Slider(3, 15, 3, step=2,
                                            label="Kernel Size")

                # Process button
                process_btn = gr.Button("▶️ Process Image", variant="primary", size="lg")

            # ============ RIGHT COLUMN: Outputs ============
            with gr.Column(scale=2):

                # Row 1: Original / Noisy / Restored / Difference
                with gr.Row():
                    out_original = gr.Image(label="Original", type="numpy")
                    out_noisy = gr.Image(label="Noisy", type="numpy")
                with gr.Row():
                    out_restored = gr.Image(label="Restored", type="numpy")
                    out_diff = gr.Image(label="Difference Map", type="numpy")

                # Row 2: Segmentation / Morphology
                with gr.Row():
                    out_seg = gr.Image(label="Segmentation Mask", type="numpy")
                    out_morph = gr.Image(label="Morphology Result", type="numpy")

                # Metrics
                with gr.Row():
                    out_mse = gr.Textbox(label="MSE", interactive=False)
                    out_psnr = gr.Textbox(label="PSNR (dB)", interactive=False)

                # Feature table
                out_features = gr.Dataframe(
                    headers=["Feature", "Value"],
                    label="Extracted Features",
                    interactive=False,
                )

        # ============ Event binding ============
        all_inputs = [
            image_dd,
            noise_type, noise_density, noise_sigma, noise_freq, noise_amp,
            noise_gshape, noise_gscale, noise_ulow, noise_uhigh,
            spatial_filter, spatial_ksize, gauss_sigma,
            freq_filter, fft_d0, notch_cf, notch_bw,
            hist_eq, clahe_flag,
            seg_method, global_thresh, adapt_block, adapt_c,
            morph_op, morph_kshape, morph_ksize,
        ]

        all_outputs = [
            out_original, out_noisy, out_restored, out_diff,
            out_seg, out_morph,
            out_mse, out_psnr,
            out_features,
        ]

        process_btn.click(fn=process_image, inputs=all_inputs, outputs=all_outputs)

    return app


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    print("\nLaunching Gradio interface...")
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )
