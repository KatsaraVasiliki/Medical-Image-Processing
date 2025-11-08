# -*- coding: utf-8 -*-
"""
Created on Mon Mar 6 2023
@author: vikir
"""
"""
Image Display Utilities and Demonstrations
Created on Mon Mar 6 2023
@author: vikir

This script contains utility functions and demonstrations for displaying 
images and image matrices using different grayscale mapping techniques.

Supported methods:
- Simple linear mapping
- Optimal min-max normalization
- Custom threshold mapping
- Random threshold display

It works on both small matrices and real images (BMP, JPG, PNG, DICOM).
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# -------------------- UTILITY FUNCTIONS --------------------
def cls():
    """Clear terminal screen."""
    print(chr(27) + "[2J")

def RGB2GRAY(im):
    """Convert RGB image to grayscale."""
    if len(im.shape) == 3 and im.shape[2] == 3:
        Rim, Gim, Bim = im[:,:,0], im[:,:,1], im[:,:,2]
        gray = 0.2989*Rim + 0.5870*Gim + 0.1140*Bim
        return gray
    return im

# -------------------- DISPLAY FUNCTIONS --------------------
def simpleDisplay(im, image_depth, tones):
    """Simple linear mapping from image depth to gray levels."""
    im1 = np.round((tones-1) / (image_depth-1) * (im - 0))
    return im1

def optimalDisplay(im, tones):
    """Map image to tones using min-max normalization."""
    vmn, vmx = np.min(im), np.max(im)
    im1 = np.round((tones-1) * (im - vmn) / (vmx - vmn))
    return im1

def customDisplay(im, tones, V1=8, V2=28):
    """Map image to tones with custom thresholds and clipping."""
    im1 = np.round((tones-1) * (im - V1) / (V2 - V1))
    im1 = np.clip(im1, 0, tones-1)
    return im1

def thresholdDisplay(im, tones):
    """Random threshold-based display."""
    n1, n2 = random.randint(0, 255), random.randint(0, 255)
    v1, v2 = sorted([n1, n2])
    im1 = np.round((tones-1) * (im - v1) / (v2 - v1))
    im1 = np.clip(im1, 0, tones-1)
    return im1, v1, v2

# -------------------- DEMONSTRATION ON MATRIX --------------------
def demo_matrix_display():
    cls()
    m = [
        [24,14,21,27],
        [15,21,16,15],
        [13,12,26,12],
        [30,19,17,19],
    ]
    print("INITIAL IMAGE MATRIX:\n")
    im = np.asarray(m, dtype=float)
    print(im)
    
    image_depth = 32
    tones = 8
    
    print("\nSIMPLE IMAGE DISPLAY:")
    print(simpleDisplay(im, image_depth, tones))
    
    print("\nOPTIMAL IMAGE DISPLAY:")
    print(optimalDisplay(im, tones))
    
    print("\nCUSTOM IMAGE DISPLAY:")
    print(customDisplay(im, tones))
    
# -------------------- DEMONSTRATION ON REAL IMAGE --------------------
def demo_real_image(image_path):
    cls()
    
    im = plt.imread(image_path)
    if len(im.shape) == 3:
        im = RGB2GRAY(im)
    
    # Normalize to 0-255
    im = 255 * im / np.max(im) if np.max(im) > 0 else im
    im = np.asarray(im, dtype=float)
    
    image_depth = 255
    tones = 256
    
    im1 = simpleDisplay(im, image_depth, tones)
    im2 = optimalDisplay(im, tones)
    im3, v1, v2 = thresholdDisplay(im, tones)
    
    fz = 15
    plt.figure(figsize=(fz, fz))
    plt.subplot(4,1,1)
    plt.imshow(im1, cmap='gray', vmin=0, vmax=tones)
    plt.axis('off'); plt.title('Simple Display')
    
    plt.subplot(4,1,2)
    plt.imshow(im, cmap='gray', vmin=np.min(im), vmax=np.max(im))
    plt.axis('off'); plt.title('Initial Image')
    
    plt.subplot(4,1,3)
    plt.imshow(im2, cmap='gray', vmin=0, vmax=tones)
    plt.axis('off'); plt.title('Optimal Display')
    
    plt.subplot(4,1,4)
    plt.imshow(im3, cmap='gray', vmin=0, vmax=tones)
    plt.axis('off'); plt.title(f'Threshold Display (v1={v1}, v2={v2})')
    plt.show()

# -------------------- MAIN PROGRAM --------------------
if __name__ == "__main__":
    demo_matrix_display()
    
    # Example usage: change this path to an image in your ./images/ folder
    image_path = "./images/pelvis.bmp"
    demo_real_image(image_path)
