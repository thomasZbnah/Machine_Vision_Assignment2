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

mpl.rcParams['figure.dpi']= 150
db = np.load("1-NN-descriptor-vects.npy")

image4 = plt.imread('image.png',False)
R,C = image4.shape[0],image4.shape[1]
imutils.imshow(image4)
image = np.zeros((image4.shape[0],image4.shape[1],3),dtype = "float32")
image[...,0] = image4[...,0]
image[...,1] = image4[...,1]
image[...,2] = image4[...,2]
imutils.imshow(image)
hsv_image = color.rgb2hsv(image)
imutils.imshow(hsv_image[...,0],colourmap='Greys')
imutils.imshow(hsv_image[...,1],colourmap='Greys') #cmap='gist_gray'
imutils.imshow(hsv_image[...,2],colourmap='Greys')

mask_image = np.ones((image4.shape[0],image4.shape[1],3),dtype = "float32")
for i in range(0,R):
    for j in range(0,C):
        if (hsv_image[i,j,0] <= 0.05)and(hsv_image[i,j,1] <=0.2): #if (hsv_image[i,j,0] <= 0.05)and(hsv_image[i,j,1] <=0.05):
            mask_image[i,j,0] = 0.99
imutils.imshow(mask_image[...,0],colourmap='gist_gray')

