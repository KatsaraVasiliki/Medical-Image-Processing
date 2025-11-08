"""
Advanced Image Display and Windowing Utilities
Created on 2023
@author: vikir

This script contains functions to apply different display methods to images:
- Simple linear mapping
- Windowing techniques (simple, broken, double)
- Nonlinear transformations (inverse, logarithmic, power, sine, exponential, sigmoid, cosine)

It works on both small matrices and real images (BMP, JPG, PNG, DICOM).
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import moduleUtils as U  # Utility module with cls() function

# -------------------- UTILITY FUNCTIONS --------------------
def cls():
    """Clear terminal screen (delegates to moduleUtils)."""
    U.cls()

def RGB2GRAY(im):
    """Convert an RGB image to grayscale."""
    if len(im.shape) == 3 and im.shape[2] == 3:
        Rim, Gim, Bim = im[:,:,0], im[:,:,1], im[:,:,2]
        gray = 0.2989*Rim + 0.5870*Gim + 0.1140*Bim
        return gray
    return im

def readImage2gray(imageFile):
    """Read an image and convert to grayscale if needed."""
    imtype = str(imageFile[-3:])
    if imtype != 'dcm':
        im = plt.imread(imageFile)
    else:
        import pydicom as dicom
        im = dicom.dcmread(imageFile)
        im = np.array(im.pixel_array, int)
    if len(im.shape) == 3:
        im = RGB2GRAY(im)
    return im

def imPlot(im, title, tones, fz):
    """Plot a grayscale image using matplotlib."""
    im_pil = Image.fromarray(np.asarray(im, dtype="uint8"), "L")
    plt.imshow(im_pil, cmap=plt.cm.gray, vmin=0, vmax=tones)
    plt.title(title)
    plt.axis("off")

# -------------------- DISPLAY FUNCTIONS --------------------
def simpleDisplay(im, image_depth, tones):
    """Linear mapping from image depth to gray levels."""
    return np.round((tones-1)/(image_depth-1) * im)

def simpleWindow(im, wc, ww, image_depth, tones):
    """Simple windowing method."""
    im1 = np.zeros(im.shape, dtype=float)
    Vb = min((2.0*wc + ww)/2.0, image_depth)
    Va = max(Vb - ww, 0)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            Vm = im[i,j]
            if Vm < Va:
                t = 0
            elif Vm > Vb:
                t = tones - 1
            else:
                t = (tones-1)*(Vm - Va)/(Vb - Va)
            im1[i,j] = np.round(t)
    return im1

def brokenWindow(im, image_depth, tones, gray_val, im_val):
    """Broken-window mapping."""
    im = np.asarray(im, dtype=float)
    im1 = np.zeros(im.shape, dtype=float)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j] <= im_val:
                im1[i,j] = gray_val / im_val * im[i,j]
            else:
                im1[i,j] = ((tones-1)-(gray_val+1)) / (image_depth-(im_val+1)) * (im[i,j]- (im_val+1)) + (gray_val+1)
    return np.round(im1)

def doubleWindow(im, ww1, wl1, ww2, wl2, image_depth, tones):
    """Double-window mapping."""
    im = np.asarray(im, dtype=float)
    im1 = np.zeros(im.shape, dtype=float)
    half = (tones/2) - 1
    ve1 = round((2.0*wl1 + ww1)/2.0); vs1 = ve1 - ww1
    ve2 = round((2.0*wl2 + ww2)/2.0); vs2 = ve2 - ww2
    if vs2 < ve1:
        new_point = round((vs2+ve1)/2.0)
        ve1 = new_point
        vs2 = ve1
    vs1 = max(vs1, 0)
    ve2 = min(ve2, image_depth)
    
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            val = im[i,j]
            if val < vs1:
                im1[i,j] = 0
            elif vs1 <= val <= ve1:
                im1[i,j] = round((half)/(ve1-vs1)*(val-vs1))
            elif ve1 < val < vs2:
                im1[i,j] = half + 1
            elif vs2 <= val <= ve2:
                im1[i,j] = round(((tones-1)-(half+1))/(ve2-vs2)*(val-vs2) + (half+1))
            else:
                im1[i,j] = tones-1
    return im1

def formPlotFunction(tones, Choice):
    """Generate various nonlinear display functions."""
    import math
    w = np.zeros(tones)
    text = ''
    for i in range(tones):
        if Choice == 0:  # inverse
            w[i] = tones - i - 1
            text = 'inverse'
        elif Choice == 1:  # logarithmic
            r = 0.05
            w[i] = math.log(1 + r*i)
            text = 'logarithmic'
        elif Choice == 2:  # inverse logarithmic
            c = 128
            w[i] = np.exp(i)**(1/c) - 1
            text = 'inverse logarithmic'
        elif Choice == 3:  # power
            gamma = 0.55
            w[i] = i**gamma
            text = 'power'
        elif Choice == 4:  # sine window
            w[i] = np.sin(2*np.pi*i/(4*(tones-1)))
            text = 'sine-window'
        elif Choice == 5:  # exponential
            w[i] = 1 - np.exp(-i/90)
            text = 'exp-window'
        elif Choice == 6:  # sigmoid
            w[i] = 1 / (1 + np.exp(-i/70))
            text = 'sigmoid-window'
        elif Choice == 7:  # cosine
            w[i] = np.cos(2*np.pi*i/(4*(tones-1)))
            text = 'cosine-window'
    w = (tones-1) * ((w - np.min(w)) / (np.max(w) - np.min(w)))
    return w, text

# -------------------- MAIN PROGRAM --------------------
if __name__ == "__main__":
    U.cls()
    tones = 256
    image_depth = 255
    
    # Image selection
    path = "./images/"
    imageNames = ["head1.bmp", "head5.bmp", "head6.bmp", "image2.bmp",
                  "chest.bmp", "pelvis.bmp", "AA1a.bmp", "Lung_130.bmp", "lungs.bmp"]
    print("Images: 0=head1, 1=head5, 2=head6, 3=image2, 4=chest, 5=pelvis, 6=AA1a, 7=Lung_130, 8=lungs")
    n = int(input("Enter the number of the image(0-8) you want to select: "))
    while n<0 or n>8:
        n = int(input("Invalid choice. Enter 0-8: "))
    imageFile = path + imageNames[n]
    
    # Read grayscale image
    im = readImage2gray(imageFile)
    
    # Apply all display functions
    for fChoice in range(0,12):
        im = np.asarray(im, float)
        w, sText = formPlotFunction(tones, fChoice)
        
        if fChoice < 9:
            # Normalize image to 0..tones-1
            mn, mx = np.min(im), np.max(im)
            im1 = np.round((tones-1)*(im-mn)/(mx-mn))
            for i in range(im.shape[0]):
                for j in range(im.shape[1]):
                    v = int(im1[i,j])
                    im1[i,j] = np.int32(w[v])
        elif fChoice == 9:
            im1 = simpleWindow(im, wc=50, ww=250, image_depth=image_depth, tones=tones)
        elif fChoice == 10:
            im1 = brokenWindow(im, image_depth, tones, gray_val=128, im_val=70)
        elif fChoice == 11:
            im1 = doubleWindow(im, ww1=100, wl1=50, ww2=100, wl2=150, image_depth=image_depth, tones=tones)
        
        # Display original and transformed images
        plt.figure(figsize=(12,12))
        plt.subplot(1,2,1)
        imPlot(im, "Initial Image", tones, 12)
        plt.subplot(1,2,2)
        imPlot(im1, f"Display Method: {sText}", tones, 12)
        plt.show()
