# Advanced Digital Image Processing — Project Report

## 1. Introduction

This report presents a complete digital image processing pipeline applied to 5 synthetic object images. The pipeline covers noise injection, spatial and frequency-domain restoration, segmentation, morphological processing, feature extraction, and classification using PCA and Minimum Distance methods.

## 2. Dataset Description

| # | Object | Geometric Profile | Description |
|---|--------|-------------------|-------------|
| 1 | Circle | Circular | A filled circle with a single smooth boundary and constant curvature. Useful for... |
| 2 | Wrench | Elongated | An elongated tool shape with high aspect ratio and a mix of straight and curved ... |
| 3 | Hexagon | Polygonal | A regular six-sided polygon with straight edges meeting at 120-degree angles. Us... |
| 4 | Leaf | Irregular | An organic leaf shape with asymmetric curves and no straight edges. Useful for t... |
| 5 | Washer | Hollow | A ring (donut) shape with a center hole and 4 bolt holes. Useful for testing hol... |

## 3. Per-Image Analysis

### 3.1. Circle (Circular)

**Noise type:** salt_pepper (density=0.05)

**Best filter:** Median 3x3

**MSE:** 10.42 | **PSNR:** 37.95 dB

**Best segmentation:** Global (IoU = 0.9994)

**Best morphology:** Closing (ellipse 7×7) IoU = 0.9995

![Circle Comparison](output\figures\circle_comparison.png)

![Circle Segmentation & Morphology](output\figures\circle_seg_morph.png)

#### Parameter Justification

- **Median 3x3** was selected because median filtering is the most effective method for impulse-type noise, as it replaces outlier pixels with the neighborhood median without blurring edges.
- **Global thresholding** performed best due to uniform illumination and clear intensity separation between object and background.
- **Closing (ellipse 7×7)** was chosen because it fills small gaps and holes in the segmentation mask, producing a more complete object representation.

### 3.2. Wrench (Elongated)

**Noise type:** gaussian (mean=0, sigma=25)

**Best filter:** Median 3x3

**MSE:** 56.08 | **PSNR:** 30.64 dB

**Best segmentation:** Global (IoU = 0.9991)

**Best morphology:** Closing (rect 3×3) IoU = 0.9991

![Wrench Comparison](output\figures\wrench_comparison.png)

![Wrench Segmentation & Morphology](output\figures\wrench_seg_morph.png)

#### Parameter Justification

- **Median 3x3** was selected because median filtering is the most effective method for impulse-type noise, as it replaces outlier pixels with the neighborhood median without blurring edges.
- **Global thresholding** performed best due to uniform illumination and clear intensity separation between object and background.
- **Closing (rect 3×3)** was chosen because it fills small gaps and holes in the segmentation mask, producing a more complete object representation.

### 3.3. Hexagon (Polygonal)

**Noise type:** periodic (frequency=30, amplitude=40)

**Best filter:** Notch BW=5

**MSE:** 190.89 | **PSNR:** 25.32 dB

**Best segmentation:** Global (IoU = 1.0000)

**Best morphology:** Opening (ellipse 3×3) IoU = 1.0000

![Hexagon Comparison](output\figures\hexagon_comparison.png)

![Hexagon Segmentation & Morphology](output\figures\hexagon_seg_morph.png)

#### Parameter Justification

- **Notch BW=5** was selected because the periodic noise creates distinct peaks in the frequency spectrum that can be precisely targeted and removed by a notch filter, leaving other frequencies intact.
- **Global thresholding** performed best due to uniform illumination and clear intensity separation between object and background.
- **Opening (ellipse 3×3)** was chosen because it removes small noise blobs (via erosion) then restores object size (via dilation), effectively cleaning the mask without shrinking it.

### 3.4. Leaf (Irregular)

**Noise type:** gamma (shape=5.0, scale=5.0)

**Best filter:** Median 3x3

**MSE:** 10.84 | **PSNR:** 37.78 dB

**Best segmentation:** Global (IoU = 0.9998)

**Best morphology:** Closing (rect 3×3) IoU = 0.9998

![Leaf Comparison](output\figures\leaf_comparison.png)

![Leaf Segmentation & Morphology](output\figures\leaf_seg_morph.png)

#### Parameter Justification

- **Median 3x3** was selected because median filtering is the most effective method for impulse-type noise, as it replaces outlier pixels with the neighborhood median without blurring edges.
- **Global thresholding** performed best due to uniform illumination and clear intensity separation between object and background.
- **Closing (rect 3×3)** was chosen because it fills small gaps and holes in the segmentation mask, producing a more complete object representation.

### 3.5. Washer (Hollow)

**Noise type:** uniform (low=-30, high=30)

**Best filter:** Median 3x3

**MSE:** 47.96 | **PSNR:** 31.32 dB

**Best segmentation:** Global (IoU = 0.9996)

**Best morphology:** Opening (rect 3×3) IoU = 0.9996

![Washer Comparison](output\figures\washer_comparison.png)

![Washer Segmentation & Morphology](output\figures\washer_seg_morph.png)

#### Parameter Justification

- **Median 3x3** was selected because median filtering is the most effective method for impulse-type noise, as it replaces outlier pixels with the neighborhood median without blurring edges.
- **Global thresholding** performed best due to uniform illumination and clear intensity separation between object and background.
- **Opening (rect 3×3)** was chosen because it removes small noise blobs (via erosion) then restores object size (via dilation), effectively cleaning the mask without shrinking it.

## 4. Group Summary Table

| Image | Object | Profile | Noise | Best Filter | Best Threshold | Best Morphology | PSNR (dB) | MSE | Predicted Class |
|-------|--------|---------|-------|-------------|----------------|-----------------|-----------|-----|-----------------|
| Circle | Circle | Circular | salt_pepper | Median 3x3 | Global | Closing ellipse 7x7 | 37.95 | 10.42 | Circular |
| Wrench | Wrench | Elongated | gaussian | Median 3x3 | Global | Closing rect 3x3 | 30.64 | 56.08 | Elongated |
| Hexagon | Hexagon | Polygonal | periodic | Notch BW=5 | Global | Opening ellipse 3x3 | 25.32 | 190.89 | Polygonal |
| Leaf | Leaf | Irregular | gamma | Median 3x3 | Global | Closing rect 3x3 | 37.78 | 10.84 | Irregular |
| Washer | Washer | Hollow | uniform | Median 3x3 | Global | Opening rect 3x3 | 31.32 | 47.96 | Hollow |

## 5. Feature Extraction

| Object | Area | Perimeter | Circularity | Aspect Ratio | Solidity | Extent | Holes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Circle | 70260.5 | 992.73 | 0.8959 | 1.0 | 0.996 | 0.7859 | 0 |
| Wrench | 22178.0 | 1127.35 | 0.2193 | 3.7615 | 0.5751 | 0.4963 | 0 |
| Hexagon | 66102.0 | 1003.38 | 0.8251 | 1.1588 | 0.9961 | 0.7434 | 0 |
| Leaf | 48042.0 | 881.52 | 0.7769 | 1.3377 | 0.9949 | 0.6731 | 0 |
| Washer | 79924.0 | 1056.43 | 0.8999 | 1.0 | 0.9968 | 0.7854 | 5 |

## 6. PCA Analysis

![PCA Scatter Plot](output\pca\pca_scatter.png)

The PCA scatter plot projects the 7-dimensional feature space onto 2 principal components. Points are colored by geometric class. Clear separation between classes indicates that the extracted shape features effectively distinguish different object types.

## 7. Classification Report

```
======================================================================
CLASSIFICATION REPORT - Minimum Distance Classifier
======================================================================

Method: Class centroids computed from augmented training data.
Distance metric: Euclidean distance in standardized feature space.

Object: Circle
  True class:      Circular
  Predicted class:  Circular  [CORRECT]
  Min distance:     0.139
  All distances:    Hollow: 2.6686, Elongated: 5.8544, Polygonal: 0.4898, Circular: 0.139, Irregular: 1.9854

Object: Wrench
  True class:      Elongated
  Predicted class:  Elongated  [CORRECT]
  Min distance:     0.2567
  All distances:    Hollow: 6.3871, Elongated: 0.2567, Polygonal: 5.3382, Circular: 5.6511, Irregular: 5.0091

Object: Hexagon
  True class:      Polygonal
  Predicted class:  Polygonal  [CORRECT]
  Min distance:     0.1449
  All distances:    Hollow: 2.737, Elongated: 5.396, Polygonal: 0.1449, Circular: 0.4659, Irregular: 1.6938

Object: Leaf
  True class:      Irregular
  Predicted class:  Irregular  [CORRECT]
  Min distance:     0.0984
  All distances:    Hollow: 3.7328, Elongated: 5.2249, Polygonal: 1.7222, Circular: 1.844, Irregular: 0.0984

Object: Washer
  True class:      Hollow
  Predicted class:  Hollow  [CORRECT]
  Min distance:     0.1666
  All distances:    Hollow: 0.1666, Elongated: 6.4327, Polygonal: 2.6672, Circular: 2.6415, Irregular: 3.6421

Overall Accuracy: 5/5 (100.0%)
======================================================================
```

**Overall accuracy:** 100.0%

## 8. Conclusion

This project demonstrated a complete image processing pipeline from noise injection through classification. Key findings:

- Median filtering is most effective for salt-and-pepper noise.
- Notch filtering in the frequency domain precisely removes periodic noise.
- Otsu thresholding provides robust automatic segmentation for bimodal images.
- Morphological opening/closing effectively cleans segmentation masks.
- Shape features (circularity, aspect ratio, solidity) provide strong discriminative power for geometric classification.
