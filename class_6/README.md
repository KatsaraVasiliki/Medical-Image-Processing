# Radon Transform & Image Reconstruction

## Overview
This project implements **tomographic image reconstruction** using the **Radon Transform**, a mathematical model that simulates the data acquisition process in **Computed Tomography (CT)** scanners.

It allows you to:
- Generate **sinograms** (projections of an image at multiple angles)
- Reconstruct the original image using:
  - **Filtered Back Projection (FBP)**
  - **Algebraic Reconstruction Technique (ART / SART)**
- Evaluate reconstruction quality using a **Structural Error (ST_ERROR)** metric.

The project also uses an auxiliary module — `moduleImRestorationFunctions.py` — which provides common image restoration utilities such as normalization, windowing, Gaussian filtering, and the error metric computation.

---

## Key Features

- **Radon Transform** for simulating tomographic projections
- **Reconstruction methods**:
  - **Filtered Back Projection (FBP)** with selectable filters:
    - `ramp`, `shepp-logan`, `cosine`, `hamming`, `hann`
  - **Algebraic Reconstruction Technique (ART / SART)** with adjustable iterations
- **Image preprocessing**:
  - Grayscale conversion
  - Normalization to 8-bit intensity range [0–255]
- **Automatic cropping** to remove black borders from reconstructed images
- **Structural Error (ST_ERROR)** metric for quantitative quality assessment
- **Visualization**:
  - Original image
  - Sinogram
  - Reconstructed and cropped image with error and filter information

