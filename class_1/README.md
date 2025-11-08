# Image Display Utilities

This repository contains Python code for displaying images and image matrices with various visualization methods. The code supports:

- Simple linear mapping to grayscale.
- Optimal min-max scaling.
- Custom threshold mapping.
- Random threshold-based display.

It works with both **numerical matrices** and **real images** (BMP, JPG, PNG, DICOM, etc.). Color images are automatically converted to grayscale.

## Features

1. `simpleDisplay(im, image_depth, tones)`: Simple mapping of pixel values to a fixed number of gray tones.
2. `optimalDisplay(im, tones)`: Optimized display using min-max normalization.
3. `customDisplay(im, tones, V1=8, V2=28)`: Custom thresholding and clipping.
4. `thresholdDisplay(im, tones)`: Random thresholding within 0-255 range.
5. `RGB2GRAY(im)`: Converts RGB images to grayscale.
6. Demo functions for **matrix examples** and **real image visualization**.

## Requirements

- Python 3.x  
- Numpy  
- Matplotlib  
- Pillow (PIL)  
- Optional: `pydicom` for DICOM images (`pip install pydicom`)


