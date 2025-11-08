## Histogram & Equalization Utilities

This module provides functions to apply histogram-based transformations and equalization methods to images, including:

- Cumulative Distribution Function (CDF) based equalization
- Histogram Equalization (HE)
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- OpenCV equalizeHist and CLAHE methods
- Linear and Nonlinear windowing
- Histogram plotting and normalization

### Features

1. `histCumsum(im)`: Compute the cumulative histogram of an image.  
2. `imNormalize(im, tones)`: Normalize a 2D array to 0..tones-1.  
3. `f_histogram(A, image_depth, tones)`: Compute histogram of image values.  
4. `f_hequalization(A, image_depth, tones)`: Implement CDF-based histogram equalization.  
5. `apply_linear_windows(im, tones, image_depth, vals, Choice)`: Apply linear windowing.  
6. `apply_nonlinear_window_function(im, tones, Choice)`: Apply nonlinear windowing.  
7. `apply_non_linear_window_to_image(im, w, tones)`: Apply custom nonlinear mapping.  

### Requirements

- Python 3.x  
- Numpy, Matplotlib, OpenCV (`pip install opencv-python`)  
- Optional: `scikit-image` for advanced equalization methods (`pip install scikit-image`)  
- Optional: `pydicom` for DICOM images  

### Usage

The script allows you to select an image, choose a windowing function, and then apply one of the histogram equalization methods.  


