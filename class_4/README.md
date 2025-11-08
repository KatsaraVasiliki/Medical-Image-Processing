# Unified Image Processing Script

## Description
This Python script demonstrates basic image processing techniques using 2D convolution and filtering. 
It provides two functionalities:

1. **Small Matrix Example**: Processes a 4x4 example matrix using smoothing, Laplacian, high-emphasis, and median filters. Computes the noise difference to classify the filter as low-pass or high-pass.

2. **Full Image Processing**: 
   - Loads BMP images from the `./images/` folder.
   - Converts RGB images to grayscale.
   - Applies one of the four enhancement methods interactively:
     - Smoothing (low-pass)
     - Laplacian (high-pass)
     - High-emphasis (sharpening)
     - Median (noise reduction)
   - Displays the original and processed images.

## Requirements
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- PIL (Pillow)

## Usage
1. Place your images in a folder called `images/`.

