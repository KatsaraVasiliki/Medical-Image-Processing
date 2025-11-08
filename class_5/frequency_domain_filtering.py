# -*- coding: utf-8 -*-
"""
Merged Script: Frequency Domain Filtering & Image Restoration

Created on: Sat Apr 1 2023

Description:
This Python script demonstrates **1-D and 2-D frequency domain filtering** 
and **image restoration techniques** using Butterworth, Gaussian, and Exponential filters. 

Main functionalities:
1. **Frequency domain filtering**:
   - Butterworth High-Pass (HP) and Band-Pass (BP) filters.
   - Gaussian and Exponential filters.
   - 1-D filter design converted to 2-D filter for image processing.
2. **Image restoration**:
   - Generalized Wiener filters (Inverse, Wiener, Power).
   - scikit-image deconvolution methods (Wiener, Unsupervised Wiener, Richardson-Lucy).
3. **Visualization**:
   - Original, filtered, and restored images.
   - Frequency response plots for 1-D and 2-D filters.

This script is useful for experimenting with spatial frequency filtering, 
image enhancement, and restoration in the frequency domain.

"""

import numpy as np
import matplotlib.pyplot as plt
import moduleUtils as U
import moduleFreqEnhancementFunctions as en
import FreqDom_1d_Filters as Fil
import moduleImRestorationFunctions as iR

# ----------------------- Butterworth Filters -------------------------
def ButterworthHP(N, ndegree, fco, trans):
    fh = np.zeros(np.int32(N), dtype=float)
    if (N % 2) == 0:
        L = np.round(N / 2 + 1)
        M = np.round(N / 2 + 2)
    else:
        L = np.round(N / 2 + 0.5)
        M = np.round(N / 2 + 1 + 0.5)

    for k in range(np.int32(L)):
        fh[k] = 1.0 / (1.0 + np.power((2 * fco / (3 * k + 0.001)), 2 * ndegree))

    for k in range(np.int32(L)):
        if k < int(N / 2 - trans):
            fh[k] = fh[k + int(trans)]
        else:
            fh[k] = fh[int(N / 2)]

    sText = 'Butterworth HP'

    for k in range(np.int32(M - 1), np.int32(N)):
        fh[k] = fh[np.int32(N - k)]

    return fh / np.max(fh), sText


def ButterworthBP(N, ndegree, fco, trans):
    fh = np.zeros(np.int32(N), dtype=float)
    if (N % 2) == 0:
        L = np.round(N / 2 + 1)
        M = np.round(N / 2 + 2)
    else:
        L = np.round(N / 2 + 0.5)
        M = np.round(N / 2 + 1 + 0.5)

    for k in range(np.int32(L)):
        d = trans
        fh[k] = 1.0 / (1.0 + np.power(((k - d) / fco), 2 * ndegree))

    sText = 'Butterworth BP'

    for k in range(np.int32(M - 1), np.int32(N)):
        fh[k] = fh[np.int32(N - k)]

    return fh / np.max(fh), sText

# -------------------- Generalized Wiener Filter ---------------------
def generalizedWienerFilter(fh, filtType, SIGMA, im, im1):
    N = len(fh)
    C = 0.1 / SIGMA

    if filtType == 1:
        a, b = 1, 0  # Inverse Filter
        sText = "Inverse Filter"
    elif filtType == 2:
        a, b = 0, 1
        sText = "Wiener Filter"
    elif filtType == 3:
        a, b = 0.5, 1
        sText = "Power Filter"

    fhh = np.zeros(np.int32(N), dtype=float)
    for k in range(np.int32(N)):
        fhh[k] = (fh[k] ** 2 / ((fh[k] ** 2 + b * C))) ** (1 - a) / fh[k]
        if fhh[k] < C:
            fhh[k] = (fh[k] ** 2 / ((fh[k] ** 2 + b * C))) ** (1 - a) / C

    fhh = fhh / np.max(fhh)
    return fhh, sText

# -------------------- Scikit Deconvolution Filters ------------------
def ScikitDeconvolutionFilters(im, im1):
    FH = iR.from1dTo2dFilter(im, fh)
    wc, ww = 130, 256
    PSF = np.fft.fftshift(np.abs(np.fft.ifft2(FH)))
    PSF = PSF / np.max(PSF)

    from skimage import restoration as re
    wiener_im1 = re.wiener(im1, PSF, 0.001, reg=None, is_real=True, clip=False)
    wiener_im1 = iR.imNormalize(wiener_im1, tones)
    wiener_im1 = iR.simpleWindow(wiener_im1, wc, ww, 255, 256)

    im2_unsupervised_w, _ = re.unsupervised_wiener(im1, PSF, clip=False)
    im2_unsupervised_w = iR.imNormalize(im2_unsupervised_w, tones)
    im2_unsupervised_w = iR.simpleWindow(im2_unsupervised_w, wc, ww, 255, 256)

    im3_rich_lucy = re.richardson_lucy(im1, PSF, num_iter=3, clip=False)
    im3_rich_lucy = iR.imNormalize(im3_rich_lucy, tones)
    im3_rich_lucy = iR.simpleWindow(im3_rich_lucy, 60, ww, 255, 256)

    return wiener_im1, im3_rich_lucy, im2_unsupervised_w

# -------------------- Deconvolution in Frequency Domain --------------
def DeconvolveImage(im, FH):
    Fim = np.fft.fft2(im)
    Fim1 = Fim * np.fft.fftshift(FH)
    im1 = np.real(np.fft.ifft2(Fim1))
    return im1

# -------------------- MAIN PROGRAM ----------------------------------
U.cls()
path = "./images/"
imageNames = ["head1.bmp", "head5.bmp", "head6.bmp", "AA1a.bmp",
              "Angio.jpg", "head8.bmp", "lung_130.bmp", "foot.bmp"]

iFile = 4
F = imageNames[int(iFile)]
print('Chosen file:', F)
imageFile = path + F

im = en.loadImage(imageFile)
im = np.asarray(im, dtype=float)
im = en.imNormalize(im, 256)

# Filter parameters
tones = 256
TYPE = 1  # 1-Butterworth, 2-Gaussian, 3-Exponential
FILTER = 2  # 1-LP, 2-HP, 3-BR, 4-BP
restoreType = 1  # 1-Inverse, 2-Wiener, 3-Power, 4+ scikit
SIGMA = np.std(im)

M, N = np.size(im, 0), np.size(im, 1)
Flength = np.round(np.sqrt(M * M + N * N))

# Example: Butterworth HP and BP filtering
fh1, sText1 = ButterworthHP(Flength, 1, round(Flength * 0.2), round(Flength * 0.01))
fh2, sText2 = ButterworthBP(Flength, 1, round(Flength * 0.4), round(Flength * 0.1))

# Convert 1-D filter to 2-D
FH1 = en.design2dFilter(im, fh1)
FH2 = en.design2dFilter(im, fh2)

# Apply filters
im1 = en.filterImage(im, FH1, tones)
im2 = en.filterImage(im, FH2, tones)
im1 = en.imNormalize(im1, tones)
im2 = en.imNormalize(im2, tones)

# Optional restoration
if restoreType < 4:
    fhh, sText = generalizedWienerFilter(fh1, restoreType, SIGMA, im, im1)
    FHH = iR.from1dTo2dFilter(im1, fhh)
    im_restored = DeconvolveImage(im1, FHH)
    im_restored = en.imNormalize(im_restored, tones)

