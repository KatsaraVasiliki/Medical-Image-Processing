# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:05:26 2023
Author: vikir

Introduction:
This Python script demonstrates basic image processing techniques using 
2D convolution and filtering methods. It provides two main functionalities:

1. **Small matrix demonstration**: Applies smoothing, Laplacian, high-emphasis,
   and median filters to a 4x4 example matrix and calculates noise difference.

2. **Full image processing**: Loads an image from the './images/' folder 
   (BMP format), converts it to grayscale if needed, and applies a selected 
   image enhancement method interactively. Supported methods include:
   - Smoothing (low-pass filter)
   - Laplacian (high-pass filter)
   - High-emphasis (sharpening)
   - Median (noise reduction)

"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image

# ---------------- Utilities ----------------
def cls():
    """Clear console."""
    print(chr(27) + "[2J") 

def conv2(im, mask):
    """2D convolution using scipy.signal.convolve2d"""
    return signal.convolve2d(im, mask, mode='same')

def MeanAndStd(im, im2):
    """Compute mean and std deviation for central part of images."""
    M, N = im.shape
    central_im = im[1:-1, 1:-1]
    central_im2 = im2[1:-1, 1:-1]
    mean_central_im = np.mean(central_im)
    std_central_im = np.std(central_im, ddof=1)
    mean_central_im2 = np.mean(central_im2)
    std_central_im2 = np.std(central_im2, ddof=1)
    return mean_central_im, std_central_im, mean_central_im2, std_central_im2

# ---------------- Filtering Methods ----------------
def smoothing_method(im):
    kernel = np.ones((3,3))
    kernel /= np.sum(kernel)
    im_conv = conv2(im, kernel)
    im2 = np.asarray(im, dtype=float)
    im2[1:-1,1:-1] = im_conv[1:-1,1:-1]
    im2 = np.around(im2)
    noise_diff = 100*(np.std(im[1:-1,1:-1], ddof=1) - np.std(im2[1:-1,1:-1], ddof=1)) / np.std(im[1:-1,1:-1], ddof=1)
    return im2, noise_diff

def high_emphasis_method(im, choose_kernel=1, image_depth=255):
    if choose_kernel == 0:
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    elif choose_kernel == 1:
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    elif choose_kernel == 2:
        kernel = np.array([[-1,-2,-1],[-2,13,-2],[-1,-2,-1]])
    else:
        center=15; cross=-2; corner=-1
        kernel = np.array([[corner,cross,corner],[cross,center,cross],[corner,cross,corner]])
    kernel /= np.sum(kernel) if np.sum(kernel) > 0 else 1
    im_conv = conv2(im, kernel)
    im2 = np.asarray(im, dtype=float)
    im2[1:-1,1:-1] = np.clip(np.round(im_conv[1:-1,1:-1]), 0, image_depth)
    noise_diff = 100*(np.std(im[1:-1,1:-1], ddof=1) - np.std(im2[1:-1,1:-1], ddof=1)) / np.std(im[1:-1,1:-1], ddof=1)
    return im2, noise_diff

def laplacian_method(im, image_depth=255):
    kernel = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    im_conv = conv2(im, kernel)
    im2 = np.asarray(im, dtype=float)
    im2[1:-1,1:-1] = np.clip(np.round(im_conv[1:-1,1:-1]), 0, image_depth)
    noise_diff = 100*(np.std(im[1:-1,1:-1], ddof=1) - np.std(im2[1:-1,1:-1], ddof=1)) / np.std(im[1:-1,1:-1], ddof=1)
    return im2, noise_diff

def median_method(im):
    im_conv = signal.medfilt2d(im, (3,3))
    im2 = np.asarray(im, dtype=float)
    im2[1:-1,1:-1] = np.round(im_conv[1:-1,1:-1])
    noise_diff = 100*(np.std(im[1:-1,1:-1], ddof=1) - np.std(im2[1:-1,1:-1], ddof=1)) / np.std(im[1:-1,1:-1], ddof=1)
    return im2, noise_diff

# ---------------- Image Utilities ----------------
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def imPlot(im, title, fz=6):
    plt.figure(figsize=(fz,fz))
    plt.imshow(Image.fromarray(im.astype(np.uint8), "L"), cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis('off')
    plt.show()

def imNormalize(w, tones=256):
    w = (tones-1)*(w - np.min(w)) / (np.max(w) - np.min(w))
    return np.round(w)

# ---------------- MAIN: Small Matrix Example ----------------
cls()
m = np.array([[20, 27, 24, 13],
              [17, 10, 27, 20],
              [23, 30, 22, 24],
              [29, 31, 15, 13]], dtype=float)

image_depth = 31
fl_method = ['smoothing', 'laplacian', 'high-emphasis', 'median']

print("Initial 4x4 matrix:\n", m)
for i, method in enumerate(fl_method):
    if i == 0:
        im_processed, noise_diff = smoothing_method(m)
    elif i == 1:
        im_processed, noise_diff = laplacian_method(m, image_depth)
    elif i == 2:
        im_processed, noise_diff = high_emphasis_method(m, choose_kernel=1, image_depth=image_depth)
    elif i == 3:
        im_processed, noise_diff = median_method(m)
    print(f"\nMethod: {method}")
    print("Processed image:\n", im_processed)
    print(f"Noise difference: {noise_diff:.3f}%")
    print("LowPass filter" if noise_diff>0 else "HighPass filter")

# ---------------- MAIN: Interactive Image Example ----------------
cls()
path = "./images/"
imageNames = ["AA1a.bmp","brain_CT.bmp","brain_CT_noisy.bmp","chest.bmp",
              "HEAD1.BMP","HEAD5.bmp","lung_130.bmp","HEAD6.bmp","Image2.bmp","lungs.bmp","Pelvis.bmp"]
iFile = int(input("Choose image number 0-10 (default 10): ") or 10)
imageFile = path + imageNames[iFile]
im = plt.imread(imageFile).astype(float)
if len(im.shape) == 3:
    im = rgb2gray(im)

fl_method = ['smoothing', 'laplacian', 'high-emphasis', 'median']
choose_method = int(input("Choose method (0:smoothing,1:laplacian,2:high-emphasis,3:median, default 2): ") or 2)

if choose_method == 0:
    im_processed, _ = smoothing_method(im)
elif choose_method == 1:
    im_processed, _ = laplacian_method(im)
elif choose_method == 2:
    choose_kernel = int(input("Choose high-emphasis kernel 0-3 (default 1): ") or 1)
    im_processed, _ = high_emphasis_method(im, choose_kernel)
elif choose_method == 3:
    im_processed, _ = median_method(im)

imPlot(im, "Original Image")
imPlot(im_processed, f"Processed Image ({fl_method[choose_method]})")
