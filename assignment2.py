# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 15:16:46 2022

@author: thoma
"""

import numpy as np 
import imutils
import matplotlib as mpl
from skimage.transform import rotate ## Image rotation routine
import skimage.color as color
import scipy.fftpack as fft          ## Fast Fourier Transform
import scipy.misc                    ## Save images
import imageio
import matplotlib.pyplot as plt
from numba import njit    # Have to import the "@njit" decorator from the Numba library.
from scipy import ndimage, misc

mpl.rcParams['figure.dpi']= 50
db = np.load("1-NN-descriptor-vects.npy")


def otsu(image):
    """Otsu thresholding of an 8-bit greyscale image.  Input must be an 8-bit
       greyscale image, returns Otsu threshold, k, and inter-class variance of 
       the threshold, sB2."""
    assert image.dtype == "uint8", "8-bit greyscale image expected."
    imsize = float(image.shape[0] * image.shape[1])
    #
    # Bincount is used to build the image histogram.  Once the histogram is
    # build, the Otsu code works exclusively with it, and doesn't need to 
    # touch the source image again.  Dividing by "imsize" turns this into
    # a probability mass function.
    #
    p = np.bincount(image.ravel(),minlength=256) / imsize
    #
    # The Otsu sweep over the histogram calculates the cumulative 
    # probability mass and cumulative means.
    #
    P1, mu = np.cumsum(p), np.cumsum(range(256)*p)
    #
    # The second Otsu sweep calculates the inter-class variance for 
    # each possible greylevel (using forms A and B), and keeps track
    # of the maximum (and its index).
    #
    muG, k_max, sB2_max = mu[-1], -1, -1.0
    for k in range(255):
        P1_k = P1[k]
        if P1_k == 0.0 or P1_k == 1.0: sB2_k = 0.0  ## Form A
        else:                                       ## Form B
            sB2_k = (mu[k] - P1_k*muG)**2 / (P1_k * (1.0 - P1_k))
        if sB2_k > sB2_max:  k_max, sB2_max = k, sB2_k
    return k_max, sB2_max

def label_regions(im):
    """Accept a binary image and return a label image corresponding to it.
       This is segmentation.  Labels are 0 for background (black in original)
       and integer values >= 1 for non-zero 4-connected regions.  
       
       Expects a binary image as 2-d array of unsigned 8-bit integers, returns
       a 2-d array of unsigned 16-bit integers containing the labels.
       
       Uses the simple recursive region growing algorithm from
       Snyder & Qi "Machine Vision", pp 187.  
    """
    R,C = im.shape
    ## L is the "labels" image, same shape as input. 
    L = np.zeros((R,C),dtype='uint16') 
    curr_label = 1
    for r in range(R):
        for c in range(C):
            if im[r,c] != 0 and L[r,c] == 0:  ## Image pixel is foreground but not yet labeled.
                curr_label = grow_region(im, L, r, c, curr_label)
    return L

## 
## "grow_region" does the job of filling a single region, using the stack-based algorithm
## described in the lectures.  
##

def grow_region(im, L, r, c, curr_label):
    R,C = im.shape
    stack = [(r,c)]                ## Create stack with (r,c) co-ordinates of current pixel.
    while len(stack) > 0:
        r,c = stack.pop()
        L[r,c] = curr_label        ## Label pixel and push 4-neighbours, if appropriate.
        rm1, rp1, cm1, cp1 = r - 1, r + 1, c - 1, c + 1
        if rm1 >= 0 and im[rm1,c] != 0 and L[rm1,c] == 0: stack.append((rm1,c))
        if rp1 < R  and im[rp1,c] != 0 and L[rp1,c] == 0: stack.append((rp1,c))
        if cm1 >= 0 and im[r,cm1] != 0 and L[r,cm1] == 0: stack.append((r,cm1))
        if cp1 < C  and im[r,cp1] != 0 and L[r,cp1] == 0: stack.append((r,cp1))
    return curr_label + 1

image4 = plt.imread('image3.png',False)
R,C = image4.shape[0],image4.shape[1]
imutils.imshow(image4)
image = np.zeros((image4.shape[0],image4.shape[1],3),dtype = "float32")
image[...,0] = image4[...,0]
image[...,1] = image4[...,1]
image[...,2] = image4[...,2]
imutils.imshow(image)
hsv_image = color.rgb2hsv(image)
imutils.imshow(hsv_image[...,0])
imutils.imshow(hsv_image[...,1],colourmap='Greys') #cmap='gist_gray'
imutils.imshow(hsv_image[...,2],colourmap='Greys')

mask_image = np.zeros((image4.shape[0],image4.shape[1]),dtype = "uint8")
for i in range(0,R):
    for j in range(0,C):
        if (hsv_image[i,j,0] >= 0.8) and (hsv_image[i,j,1] >= 0.5): #good settings
            mask_image[i,j] = 255
imutils.imshow(mask_image)

filtmasked_image = ndimage.maximum_filter(mask_image, size=5)


c_th, c_sB2 = otsu(filtmasked_image)
image_binary = filtmasked_image > c_th
image_lab = label_regions(image_binary)

# plt.figure(figsize=(10,10))
# plt.imshow(image_lab, cmap='nipy_spectral', interpolation='nearest')  # Use "interpolation='nearest'" to stop interpolation artifacts
# plt.title("Labeled Cell Image"); 
maxsize = 0
for i in range(1,np.max(image_lab)+1):
    if(np.nonzero(image_lab==i)[0].size>maxsize):
        maxsize = np.nonzero(image_lab==i)[0].size


for lab in range(1,np.max(image_lab)+1):
    signs = []
    if (np.nonzero(image_lab==lab)[0].size>maxsize*0.35):
        a = np.min(np.nonzero(image_lab==lab)[0])
        b = np.max(np.nonzero(image_lab==lab)[0])
        c = np.min(np.nonzero(image_lab==lab)[1])
        d = np.max(np.nonzero(image_lab==lab)[1])
        zero = np.zeros((b-a,d-c,3),dtype = "float32")
        for i in range(a,b):
            for j in range(c,d):
                zero[i-a,j-c,0] = image[i,j,0]
                zero[i-a,j-c,1] = image[i,j,1]
                zero[i-a,j-c,2] = image[i,j,2]
        imutils.imshow(zero)
        signs.append(zero)
    
    





















