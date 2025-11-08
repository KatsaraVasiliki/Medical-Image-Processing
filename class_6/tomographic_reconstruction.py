# -*- coding: utf-8 -*-
"""
Radon Transform & Image Reconstruction

Author: medisp-2
Created on: Sat Apr 1 2023

Description:
This script performs Radon transform-based image reconstruction using:
- Filtered Back Projection (FBP)
- Algebraic Reconstruction Technique (ART / SART)

It also computes a structural error metric (ST_ERROR) along the main diagonal
to evaluate reconstruction quality.

Dependencies:
- numpy
- matplotlib
- skimage
- OpenCV (cv2)
- moduleImRestorationFunctions as iR
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
import moduleImRestorationFunctions as iR
import cv2

#--------------------------------------------------
def cls():
    print(chr(27) + "[2J") 
#--------------------------------------------------
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
   # ------------------------------------------------
def sigNorm(x):
    xmax=np.max(x);xmin=np.min(x);
    x=(x-xmin)/(xmax-xmin)
    return(x)
#--------------------------------------------------


cls()
F=["head5.bmp","pelvis.bmp","body1.bmp"]
imageFile=F[0]
im=cv2.imread(imageFile)

im=np.asarray(im,float)


L=np.shape(im)
print(len(L))        

if(len(L)==3):
    im=rgb2gray(im)
    print('------- rgb image ----------');
A=im

max_A=np.max(A);min_A=np.min(A);
A=(A-min_A)*(255/(max_A-min_A));

N_proj=180
theta=np.arange(0,N_proj)

sinogram = radon(A, theta=theta, circle=False)

plt.figure(1)
fz=12;plt.figure(figsize=(fz*1.2,fz*0.5));
plt.subplot(1,3,1)
plt.title("Image to produce sinograms")
plt.imshow(A, cmap='gray')
plt.axis("off")
MM=np.size(A,0);NN=np.size(A,1);
plt.plot((NN,0),(MM,0),'r--', linewidth=5)

plt.subplot(1,3,2);plt.imshow(sinogram,cmap='gray',
           extent=(0, N_proj, 0, sinogram.shape[0]), aspect='auto')
plt.title('display of Sinograms ')
plt.xlabel('Angle of projection')
plt.ylabel('Tomo-projections')





##METHOD

choose_method=(input('select number regarding the method, 0:for FBP or 1:for ART... '))
if(not choose_method or int(choose_method)>1): choose_method=0;
choose_method=int(choose_method)

if choose_method==0:
    from skimage.transform import iradon
    filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
    filterChoice=(input('select number from 0 to 4 regarding the filter... '))
    if(not filterChoice or int(filterChoice)>4): filterChoice=4;
    filterChoice=int(filterChoice)
    #filterChoice=1
    
    reconstr_Image = iradon(sinogram, theta=theta, filter_name=filters[filterChoice])

    
    
else: 
    from skimage.transform import iradon_sart
    I_ART = iradon_sart(sinogram, theta=theta)
    iterations=(input('select number of iterations... '))
    if(not iterations or int(iterations)>10): iterations=1;
    iterations=int(iterations)

    for i in range(iterations):
        reconstr_Image= iradon_sart(sinogram, theta=theta,image=I_ART)

 

 ###CROPPING THE IMAGE TO DELETE BLACK FRAME       

# original image= 128x128  reconstructed=182x182 (+54x54.. 54/2=27)
# we need to delete 27 rows from the top, 27 from the bottom, 27 columns from the right and 27 from the left
crop=(reconstr_Image.shape[0] - A.shape[0]) // 2
cropped_im=reconstr_Image[crop:-crop, crop:-crop]

###---------------------------------------------------------------------------------------------

#IF I DONT WANT TO CROP THE IMAGE I CAN GET THE DIAGONAL OF THE RECONSTRUCTED IMAGE REMOVING THE ZEROS FROM START AND END

# diag=np.diag(reconstr_Image)
# d=np.nonzero(diag)
# nonzero_ind1=d[0][0]
# nonzero_ind2=d[0][-1]
# cropped_diag=diag[nonzero_ind1:nonzero_ind2+1]

###---------------------------------------------------------------------------------------------  

diag_orig=np.diag(A)#isws thelei im anti gia A 
diag_degrad=np.diag(cropped_im)
SE=iR.ST_ERROR(diag_orig,diag_degrad)

M=np.size(cropped_im,0);N=np.size(cropped_im,1);
plt.subplot(1,3,3)
plt.plot((N,0),(M,0),'r--', linewidth=5)

if choose_method==0:
    plt.subplot(1,3,3)
    plt.title("Reconstructed(cropped) Image by the FBP-method\n using filter:"
              +filters[filterChoice]+"ST_ERR= {:.2f}".format(SE))
else:
    for i in range(iterations):
        reconstr_Image= iradon_sart(sinogram, theta=theta,image=I_ART)
    
    plt.title("Reconstructed(cropped) Image by the FBP-method\n using filter:"
              +str(iterations+1)+"\n ST_ERR= {:.2f}".format(SE))

plt.imshow(cropped_im, cmap='gray')
plt.axis("off")

##SE based on parameters:
###iterations
#first SE for diag_orig=np.diag(A) 2nd for diag_orig=np.diag(im)
#1: 22.6/36.08 #2:19.23/35.8 #3=17.21/35.71 #4:15.9/_35.73 #5:15.01/35.80
#6:14.39/35.88 #7:13.95/35.98 #8:13.63/36.07 #9:13.39 #10:13.21

###filters
#A:[0':ramp' 22.31, 1:'shepp-logan':25.02, 2:'cosine':30.58, 3:'hamming':33.60, 4:'hann':34.66]
#im:[0':ramp' 34.8, 1:'shepp-logan':35.03, 2:'cosine':36.38, 3:'hamming':37.14, 4:'hann':37.57]