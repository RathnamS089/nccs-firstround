#  Mars HiRISE Anomaly Detection

### NSSC 2026 — Signal Miners

An **unsupervised anomaly-detection pipeline for Mars HiRISE orbital imagery**, combining a custom Convolutional Autoencoder with Isolation Forest to identify unusual terrain patterns and visualize regions where the model struggles to reconstruct an image.

---

## Overview

The project learns a compact representation of Mars HiRISE imagery using a **custom Convolutional Autoencoder (ConvAE)** trained from scratch. The resulting **256-dimensional latent vectors** are passed to an **Isolation Forest**, which assigns a continuous Novelty Score to each image.

A statistically justified threshold is then used to identify anomaly candidates. The highest-scoring candidates are reconstructed and visualized using reconstruction-error heatmaps.

> **Higher Novelty Score = Greater anomalousness**

### Pipeline

```text
HiRISE Image
     ↓
227 × 227 Grayscale Preprocessing
     ↓
Custom Convolutional Autoencoder
     ↓
256-D Latent Vector
     ↓
Isolation Forest
     ↓
Novelty Score
     ↓
Median + 3 × MAD Threshold
     ↓
Anomaly Candidates
     ↓
Top-5 Ranking
     ↓
Reconstruction & Error Heatmaps
```

---

## Objectives

- Learn a compact representation of normal Martian terrain.
- Detect statistically unusual HiRISE image crops.
- Generate a Novelty Score for every test image.
- Use a data-driven threshold instead of an arbitrary Top-N cutoff.
- Visualize anomalous regions through reconstruction-error heatmaps.
- Link anomalous crops back to their original HiRISE observations.
- Support further geological and imaging investigation.

---

## Dataset

The project uses the **provided HiRISE Mars Orbital Imagery dataset**.

The supplied dataset package was inspected directly and contains:

| Component | Details |
|---|---|
| Image files | **10,422 JPEG images** |
| Image dimensions | **227 × 227 pixels** |
| Image mode | **Grayscale (L)** |
| Source observations | **172 unique observations** |
| Crop mapping | **10,422 crop → source mappings** |
| Metadata | Latitude, longitude, sun angle, season, resolution |
| Seasons | N-summer, N-spring, N-autumn, N-winter |
| Resolution values | 0.25, 0.50, 1.00 |

### Dataset structure

```text
DATASETS/
├── images.zip
├── source_image_metadata.csv
├── crop_metadata_index.csv
├── Copy of source_image_metadata.csv
└── Copy of source_image_metadata(1).csv
```

The source metadata files contain information associated with the original HiRISE observations. The `crop_metadata_index.csv` file provides the mapping between individual image crops and their `source_image_id`.

This allows detected anomalies to be connected back to their original observations.

---

##  Preprocessing

Each image is prepared for the model using the following steps:

1. Convert the image to grayscale.
2. Ensure a uniform **227 × 227** resolution.
3. Convert the image into a PyTorch tensor.
4. Scale pixel values to the **[0, 1]** range.
5. Preserve the crop-to-source mapping for metadata analysis.

---

##  Model Architecture

The final model is a **custom Convolutional Autoencoder (ConvAE)** built from scratch.

It consists of:

- **Encoder** — progressively downsamples the input.
- **Latent Space** — produces a fixed **256-D vector**.
- **Decoder** — reconstructs the original 227 × 227 image.

### Encoder

| Stage | Operation | Output |
|---|---|---|
| Input | — | `1 × 227 × 227` |
| 1 | Conv2d `1→32`, k4, s2 + BatchNorm + ReLU | `32 × 113 × 113` |
| 2 | Conv2d `32→64`, k4, s2 + BatchNorm + ReLU | `64 × 56 × 56` |
| 3 | Conv2d `64→128`, k4, s2 + BatchNorm + ReLU | `128 × 28 × 28` |
| 4 | Conv2d `128→256`, k4, s2 + BatchNorm + ReLU | `256 × 14 × 14` |
| Latent | Flatten → Linear | **256-D** |

The decoder mirrors the compression process using a linear projection and four `ConvTranspose2d` stages to reconstruct the original image.

**Trainable parameters:** approximately **27.12M**

### Why 256-D?

The 256-dimensional bottleneck was selected as a balance between reconstruction quality and downstream anomaly separation.

- **128-D:** lost fine ridgeline detail.
- **256-D:** provided the reported balance.
- **512-D:** increases dimensionality and can make Isolation Forest separation harder.

---

## Training

The autoencoder is trained using a combined reconstruction and structural loss:

```text
Loss = 0.85 × MSE + 0.15 × (1 − SSIM)
```

### Training configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | ReduceLROnPlateau |
| Scheduler patience | 2 |
| Scheduler factor | 0.5 |
| Epochs | 20 |
| Batch size | 32 train / 64 evaluation |

### Documented training results

| Epoch | Train Loss | Validation Loss |
|---:|---:|---:|
| 1 | 0.0823 | 0.0630 |
| 5 | 0.0580 | 0.0564 |
| 10 | 0.0557 | 0.0546 |
| 15 | 0.0541 | 0.0509 |
| 20 | 0.0522 | 0.0503 |

The documented run shows smooth convergence, with validation loss remaining close to training loss.

---

##  Anomaly Detection

The learned 256-D latent vectors are passed to an **Isolation Forest**.

The system uses the following score convention:

> **Higher Novelty Score → More anomalous**

### Statistical Thresholding

Instead of selecting a fixed number of images, the anomaly threshold is determined from the observed Novelty Score distribution.

The documented approach is:

```text
Threshold = Median + 3 × MAD
```

where **MAD = Median Absolute Deviation**.

### Documented final-run values

| Statistic | Value |
|---|---:|
| Median | `-0.0879` |
| MAD | `0.0340` |
| Threshold | `0.0142` |
| Flagged images | `113 / 2,000` |
| Flagged rate | **5.7%** |

MAD provides a robust threshold that is less sensitive to extreme values and skewed score distributions.

### Selection logic

```text
All Test Images
      ↓
Statistical Threshold
      ↓
Anomalous Candidates
      ↓
Rank by Novelty Score
      ↓
Top-5 for Detailed Investigation
```

The Top-5 images are therefore selected **after** statistical thresholding, rather than simply declaring the five highest-scoring images to be anomalies.

---

## Reconstruction & Heatmaps

The highest-novelty candidates are passed through the decoder for visual analysis.

For each selected image, the pipeline produces:

| View | Purpose |
|---|---|
| Original | Observed HiRISE crop |
| Reconstruction | Autoencoder's learned approximation |
| Error Map | Pixel-wise reconstruction error |
| Heatmap Overlay | Highlights regions with larger error |

A localized high-error region may indicate an unusual structure, artifact, abrupt content change, or another pattern that the model has difficulty representing.

However:

> **Geological interpretations are hypotheses, not confirmed classifications.**

Diffuse reconstruction error can also occur when fine terrain texture is difficult for the current 256-D bottleneck to reconstruct.

---

## Evaluation

The project evaluates both reconstruction quality and the learned latent representation.

| Metric / Diagnostic | Purpose |
|---|---|
| **MSE** | Measures pixel-level reconstruction error |
| **SSIM** | Measures structural similarity and detail preservation |
| **ROC-AUC** | Measures normal/anomalous separation when ground-truth labels are available |
| **t-SNE / UMAP** | Qualitative diagnostic of latent-space structure |
| **Novelty Distribution** | Supports data-driven threshold selection |

The current documentation does **not** provide a final numerical ROC-AUC, accuracy, or confusion matrix for the final Isolation Forest run. No unsupported metric values are reported here.

---

## Metadata & Source Mapping

The supplied dataset includes both source metadata and crop-to-source mapping.

This enables anomaly results to be connected to:

- **Latitude**
- **Longitude**
- **Sun angle**
- **Season**
- **Resolution**
- **Source image ID**

Possible analyses include:

- Geographic distribution of anomalies.
- Comparison across acquisition seasons.
- Investigation of illumination-related effects.
- Investigation of resolution-related effects.
- Linking anomalous crops to their original HiRISE observations.

---

## Limitations

- The Novelty Score identifies unusualness but does not explain the exact cause.
- Reconstruction heatmaps support interpretation but cannot confirm geological classifications.
- The fully connected 256-D bottleneck can smooth fine dune-ripple texture.
- Some flagged images may represent texture that is difficult to reconstruct rather than genuinely novel content.
- Quantitative anomaly evaluation is limited without reliable ground-truth anomaly labels.

---

## Reproducibility

An earlier project run produced different anomaly counts because image ordering/sampling was not fully deterministic.

To ensure reproducibility:

- Use a fixed random seed.
- Sort image filenames before dataset construction.
- Keep train/validation/test splits fixed.
- Record the model version and threshold.
- Use one locked run consistently across the notebook, report, and README.

> **Important:** The final published results should come from one reproducibility-locked run.

---

## Future Work

- Use the supplied metadata for geographic and acquisition-condition analysis.
- Link Top-5 anomalies to their original source observations.
- Test alternative latent dimensions.
- Compare different reconstruction losses.
- Perform systematic architecture and loss ablation studies.
- Expand labeled evaluation where reliable anomaly labels are available.
- Improve interpretation of high-error regions using domain-informed analysis.

---

## Technologies

- **Python**
- **PyTorch**
- **NumPy**
- **Pandas**
- **PIL / OpenCV**
- **Scikit-learn**
- **Matplotlib**
- **Convolutional Autoencoder**
- **Isolation Forest**
- **MSE**
- **SSIM**
- **t-SNE**
- **UMAP**

---

## Project at a Glance

| Component | Specification |
|---|---|
| Dataset | 10,422 verified HiRISE crops |
| Input | 227 × 227 grayscale |
| Model | Custom Convolutional Autoencoder |
| Training | From scratch |
| Latent space | 256-D |
| Loss | MSE + SSIM |
| Detector | Isolation Forest |
| Score | Higher = more anomalous |
| Threshold | Median + 3 × MAD |
| Interpretability | Reconstruction error + heatmaps |
| Metadata | Crop → source image mapping |

---

## Conclusion

This project provides a from-scratch, unsupervised approach for identifying unusual patterns in Mars HiRISE imagery.

A custom convolutional autoencoder converts each image crop into a compact **256-D representation**, while Isolation Forest identifies samples that are unusual within the learned latent space.

A robust statistical threshold separates anomaly candidates, and reconstruction-error heatmaps provide an interpretable view of where the model struggles.

With the supplied source metadata and crop mapping, detected anomalies can also be traced back to their original HiRISE observations for further geographic, imaging, and scientific investigation.

---

### 👨‍💻 NSSC 2026 — Signal Miners
