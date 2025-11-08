# Image Display and Windowing Utilities

This repository contains Python code for advanced image display and windowing techniques. 
The code allows you to apply different display functions to images, including:

- Simple linear mapping
- Windowing techniques (simple, broken, double)
- Nonlinear transformations (inverse, logarithmic, power, sine, exponential, sigmoid, cosine)
- Grayscale conversion for color images

It works with both **numerical matrices** and **real images** (BMP, JPG, PNG, DICOM).

## Features

1. `simpleDisplay(im, image_depth, tones)`: Linear mapping from pixel values to gray levels.
2. `simpleWindow(im, wc, ww, image_depth, tones)`: Simple windowing method.
3. `brokenWindow(im, image_depth, tones, gray_val, im_val)`: Broken-window mapping.
4. `doubleWindow(im, ww1, wl1, ww2, wl2, image_depth, tones)`: Double-window mapping.
5. `formPlotFunction(tones, Choice)`: Generates different nonlinear display functions.
6. `RGB2GRAY(im)`: Converts RGB images to grayscale.
7. `readImage2gray(imageFile)`: Reads an image and converts it to grayscale.

## Requirements

- Python 3.x  
- Numpy  
- Matplotlib  
- Pillow (PIL)  
- Optional: `pydicom` for DICOM images (`pip install pydicom`)

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-display-utils.git

