
"""
Histogram and Equalization Utilities for Image Processing

This script allows applying various histogram-based transformations to images:
- Linear and nonlinear windowing
- Histogram equalization (CDF-HE)
- CLAHE (adaptive histogram equalization)
- OpenCV equalizeHist and CLAHE
- Histogram plotting and cumulative distribution visualization

Author: vikir
Date: 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import moduleUtils as U
import cv2 as cv
from skimage import exposure

# -------------------- UTILITY FUNCTIONS --------------------
def histCumsum(im):
    """Compute cumulative histogram (CDF) of an image."""
    hist, bins = np.histogram(im.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return cdf_normalized

def imNormalize(im, tones):
    """Normalize image data to 0..tones-1."""
    mx, mn = np.max(im), np.min(im)
    w = (tones-1) * (im - mn) / (mx - mn)
    return np.round(w)

def RGB2GRAY(im):
    """Convert an RGB image to grayscale."""
    if len(im.shape) == 3:
        R, G, B = im[:,:,0], im[:,:,1], im[:,:,2]
        gray = 0.2989*R + 0.5870*G + 0.1140*B
        return gray
    return im

def readImage2gray(imageFile):
    """Read an image file and convert to grayscale if needed."""
    imtype = imageFile[-3:]
    if imtype != 'dcm':
        im = plt.imread(imageFile)
    else:
        import pydicom as dicom
        im = dicom.dcmread(imageFile)
        im = np.array(im.pixel_array, int)
    if len(np.shape(im)) == 3:
        im = RGB2GRAY(im)
    return np.asarray(im, dtype=float)

# -------------------- HISTOGRAM & EQUALIZATION --------------------
def f_histogram(A, image_depth, tones):
    """Compute histogram of an image array with specified tones."""
    minA, maxA = 0, image_depth
    B = np.round((tones-1)*((A-minA)/(maxA-minA))) if np.max(A) > (tones-1) else A
    h = np.zeros(tones, dtype=float)
    for val in B.flatten():
        h[int(val)] += 1
    return h

def f_hequalization(A, image_depth, tones):
    """Histogram equalization based on CDF."""
    minA, maxA = 0, image_depth
    B = np.round((tones-1)*(A-minA)/(maxA-minA))
    M, N = B.shape
    Bval = B.flatten()
    p = np.argsort(Bval)
    neq = int((M*N)/tones + 0.5)
    az = int((M*N)/neq)
    zRem = int(np.remainder(M*N, neq))
    D = np.zeros(M*N)
    k = -1
    for i in range(0, neq*az, neq):
        k += 1
        for j in range(neq):
            D[i+j] = k
    for i in range(neq*az, neq*az + zRem):
        D[i] = tones - 1
    # Reassign equalized values to original positions
    L = np.zeros(M*N)
    k = -1
    for i in range(M):
        for j in range(N):
            k += 1
            L[p[k]] = D[k]
    Z = np.reshape(L, B.shape)
    Z = imNormalize(Z, tones)
    return Z

# -------------------- IMAGE DISPLAY --------------------
def imPlot(im, title, tones, fz):
    """Display grayscale image using matplotlib."""
    plt.imshow(im, cmap=plt.cm.gray, vmin=0, vmax=tones)
    plt.title(title)
    plt.axis("off")

# -------------------- WINDOWING FUNCTIONS --------------------
def apply_non_linear_window_to_image(im, w, tones):
    """Apply a nonlinear mapping to an image."""
    N, M = im.shape
    im1 = np.round((tones-1)*(im - np.min(im))/(np.max(im) - np.min(im)))
    for i in range(N):
        for j in range(M):
            im1[i,j] = int(w[int(im1[i,j])])
    return im1

def simpleWindow(im, wc, ww, image_depth, tones):
    """Simple windowing method."""
    Vb = min((2*wc + ww)/2, image_depth)
    Va = max(Vb - ww, 0)
    im1 = np.zeros_like(im, dtype=float)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            val = im[i,j]
            if val < Va: im1[i,j] = 0
            elif val > Vb: im1[i,j] = tones-1
            else: im1[i,j] = round((tones-1)*(val-Va)/(Vb-Va))
    return im1

def brokenWindow(im, image_depth, tones, gray_val, im_val):
    """Broken window mapping."""
    im1 = np.zeros_like(im, dtype=float)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j] <= im_val:
                im1[i,j] = gray_val/im_val*im[i,j]
            else:
                im1[i,j] = ((tones-1)-(gray_val+1))/(image_depth-(im_val+1))*(im[i,j]-(im_val+1)) + (gray_val+1)
    return np.round(im1)

def doubleWindow(im, ww1, wl1, ww2, wl2, image_depth, tones):
    """Double window mapping."""
    half = (tones/2)-1
    ve1 = round((2*wl1 + ww1)/2); vs1 = ve1 - ww1
    ve2 = round((2*wl2 + ww2)/2); vs2 = ve2 - ww2
    if vs2 < ve1: new_point = round((vs2 + ve1)/2); ve1 = new_point; vs2 = ve1
    vs1 = max(vs1,0); ve2 = min(ve2,image_depth)
    im1 = np.zeros_like(im, dtype=float)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            val = im[i,j]
            if val < vs1: im1[i,j] = 0
            elif vs1 <= val <= ve1: im1[i,j] = round(half*(val-vs1)/(ve1-vs1))
            elif ve1 < val < vs2: im1[i,j] = half+1
            elif vs2 <= val <= ve2: im1[i,j] = round(((tones-1)-(half+1))*(val-vs2)/(ve2-vs2) + half+1)
            else: im1[i,j] = tones-1
    return im1

def apply_linear_windows(im, tones, image_depth, vals, Choice):
    """Apply linear windowing: simple, broken, or double."""
    if Choice == 0:
        return simpleWindow(im, vals[0], vals[1], image_depth, tones), 'simple_window'
    elif Choice == 1:
        return brokenWindow(im, image_depth, tones, vals[0], vals[1]), 'broken_window'
    elif Choice == 2:
        return doubleWindow(im, vals[0], vals[1], vals[2], vals[3], image_depth, tones), 'double_window'

def apply_nonlinear_window_function(im, tones, Choice):
    """Apply nonlinear windowing based on pre-defined functions."""
    w = np.zeros(tones)
    text = ''
    for i in range(tones):
        if Choice==3: w[i], text = tones-i-1, 'inverse'
        elif Choice==4: w[i], text = np.log(1+0.05*i), 'logarithmic'
        elif Choice==5: w[i], text = np.exp(i)**(1/128)-1, 'inverse logarithmic'
        elif Choice==6: w[i], text = i**0.55, 'power'
        elif Choice==7: w[i], text = np.sin(2*np.pi*i/(4*(tones-1))), 'sine-window'
        elif Choice==8: w[i], text = 1-np.exp(-i/90), 'exp-window'
        elif Choice==9: w[i], text = 1/(1+np.exp(-i/70)), 'sigmoid'
        elif Choice==10: w[i], text = np.cos(2*np.pi*i/(4*(tones-1))), 'cosine'
        elif Choice==11: w[i], text = 1/((i+1)**0.5), 'inverse_square_root'
    w = (tones-1)*(w-np.min(w))/(np.max(w)-np.min(w))
    im1 = apply_non_linear_window_to_image(im, w, tones)
    return im1, text

# -------------------- MAIN PROGRAM --------------------
if __name__ == "__main__":
    U.cls()
    path = "./images/"
    imageNames = ["head1.bmp","head5.bmp","head6.bmp","image2.bmp","chest.bmp","pelvis.bmp","AA1a.bmp","Lung_130.bmp","lungs.bmp"]
    print("Images: 0=head1,1=head5,2=head6,...8=lungs")
    
    n = int(input("Select image (0-8): "))
    while n<0 or n>8: n=int(input("Invalid, choose 0-8: "))
    
    imageFile = path + imageNames[n]
    im = readImage2gray(imageFile)
    im = np.asarray(im, dtype=float)
    image_depth, tones = 255, 256
    
    # Select window function
    choose_window = int(input('Window function (0-11): ') or 0)
    vals = []
    if choose_window < 3:
        if choose_window==0: vals = [int(input("Center: ") or 100), int(input("Width: ") or 200)]
        elif choose_window==1: vals = [int(input("Gray_val: ") or 128), int(input("Im_val: ") or 70)]
        elif choose_window==2: vals = [int(input("WC1: ") or 50), int(input("WW1: ") or 100), int(input("WC2: ") or 200), int(input("WW2: ") or 100)]
        im1, sText = apply_linear_windows(im, tones, image_depth, vals, choose_window)
    else:
        im1, sText = apply_nonlinear_window_function(im, tones, choose_window)
    
    # Display initial and windowed image + histograms
    fz = 12
    plt.figure(figsize=(fz,fz))
    plt.subplot(3,2,1); plt.imshow(im, cmap='gray'); plt.title('Initial image'); plt.axis("off")
    plt.subplot(3,2,2); plt.hist(im.flatten(),256,[0,256], color='r'); plt.title('Initial histogram'); plt.grid(); plt.plot(histCumsum(im),'b')
    plt.subplot(3,2,3); imPlot(im1, 'Windowed: '+sText, tones, fz)
    plt.subplot(3,2,4); plt.hist(im1.flatten(),256,[0,256], color='r'); plt.title('Windowed histogram'); plt.grid(); plt.plot(histCumsum(im1),'b')

    # Choose equalization method
    methods=['CDF-HE','CLAHE','hist_equalization','equalizeHist','createCLAHE']
    choose_method = int(input('Method (0-4): ') or 0)
    
    if choose_method==0: im_eq = exposure.equalize_hist(im1); im_eq = 255*im_eq/np.max(im_eq)
    elif choose_method==1: im_eq = 255*exposure.equalize_adapthist(np.uint8(im1), clip_limit=0.03)/np.max(im1)
    elif choose_method==2: im_eq = f_hequalization(im1, image_depth, tones)
    elif choose_method==3: im_eq = cv.equalizeHist(np.uint8(cv.normalize(im1, None, 0, 255, cv.NORM_MINMAX)))
    elif choose_method==4: clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); im_eq = clahe.apply(np.uint8(cv.normalize(im1, None, 0, 255, cv.NORM_MINMAX)))
    
    plt.subplot(3,2,5); plt.imshow(im_eq, cmap='gray'); plt.title(f'Equalized: {methods[choose_method]}'); plt.axis("off")
    plt.subplot(3,2,6); plt.hist(im_eq.flatten(),256,[0,256], color='r'); plt.title('Histogram'); plt.grid(); plt.plot(histCumsum(im_eq),'b')
    plt.show()
