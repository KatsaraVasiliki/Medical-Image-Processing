# Frequency Domain Image Filtering & Restoration

## Overview
This project provides Python scripts for **image filtering and restoration** in the frequency domain. 
It includes:

- **1-D and 2-D filters**: Butterworth, Gaussian, Exponential (LP, HP, BP, BR).
- **Restoration filters**: Inverse, Wiener, Power, and scikit-image deconvolution.
- Frequency domain processing using FFT.

## Features
- Load and normalize images for processing.
- Design and apply 1-D filters converted to 2-D.
- Filter images in the frequency domain.
- Optional image restoration to remove blur/noise.
- Visualize results with plots for original, filtered, and restored images.

## Dependencies
- Python 3.x
- NumPy
- Matplotlib
- SciPy
- Pillow
- scikit-image

## Usage
1. Place images in `./images/`.
2. Set filter type (`TYPE`) and filter selection (`FILTER`).
3. Choose restoration method (`restoreType`).
4. Run the script to view results.
