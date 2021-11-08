#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:00:10 2021

@author: sachahai
"""
import matplotlib.pyplot as plt


from skimage.util import img_as_float, pad
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread, imshow, imsave
from PIL import Image
import numpy as np
import os
from skimage.util import img_as_ubyte


# image_path = "/home/sachahai/Documents/VAE_manifold/DataSets/Felix_Full_Complete/Class_1/Coverslip1_XY2_8.tiff"
image_path = "/home/sachahai/Documents/VAE_manifold/DataSets/Felix_Full_64/Class_1/Coverslip1_XY2_8.tiff"
### specify root folders
img_arr = imread(image_path)
image = Image.open(image_path)

# image_show = Image.fromarray(image_array)# .convert('CMYK')

plt.imshow(image.convert('CMYK'))
#imshow(img_arr[:,:,:3])

"""
padd to equal sizes 
"""
pad_size = max(img_arr.shape)
diff_h = pad_size - img_arr.shape[0]
diff_w = pad_size - img_arr.shape[1]
#img_padded = pad(img_arr, ((int(np.round(diff_h / 2.)), diff_h - int(np.round(diff_h / 2.))),
 #                           (int(np.round(diff_w / 2.)), diff_w - int(np.round(diff_w / 2.))), (0, 0)), constant_values=255)
img_resized = resize(img_arr, (img_arr.shape[0] // 4, img_arr.shape[1] // 4), preserve_range=False, anti_aliasing=False)

plt.imshow(img_resized[:,:,:3])
# imshow(img_resized[:,:,:3])
# imsave('/home/sachahai/Documents/VAE_manifold/DataSets/'+ 'Coverslip1_XY4_35_padded.tiff', img_padded, plugin='tifffile')
# fig, axes = plt.subplots(nrows=2, ncols=2)
# ax = axes.ravel()
### ddd

# test rescale

# firstly padding to obtain same size then rescale to target size 

'''
### read from the root folder
image1_paded = imread('/home/sachahai/Documents/VAE_manifold/DataSets/'+ 'Coverslip1_XY1_4_padded.tiff')[:, :, :3]
image1_resized = imread('/home/sachahai/Documents/VAE_manifold/DataSets/'+ 'Coverslip1_XY1_4_resized.tiff')[:, :, :3]
image2_paded = imread('/home/sachahai/Documents/VAE_manifold/DataSets/'+ 'Coverslip1_XY4_35_padded.tiff')[:, :, :3]
image2_resized = imread('/home/sachahai/Documents/VAE_manifold/DataSets/'+ 'Coverslip1_XY4_35_resized.tiff')[:, :, :3]

### specify root folders
ig, axes = plt.subplots(nrows=2, ncols=2)
ax = axes.ravel()

ax[0].imshow(image1_paded)
ax[0].set_title("Original image")
ax[0].set_xlim(0, image1_paded.shape[0])
ax[0].set_ylim(image1_paded.shape[0], 0)

ax[1].imshow(image1_resized)
ax[1].set_title("Rescaled image (aliasing)")
ax[1].set_xlim(0, image1_resized.shape[0])
ax[1].set_ylim(image1_resized.shape[0], 0)

ax[2].imshow(image2_paded)
ax[2].set_title("Resized image (no aliasing)")
ax[2].set_xlim(0, image2_paded.shape[0])
ax[2].set_ylim(image2_paded.shape[0], 0)

ax[3].imshow(image2_resized)
ax[3].set_title("Downscaled image (no aliasing)")
ax[3].set_xlim(0, image2_resized.shape[0])
ax[3].set_ylim(image2_resized.shape[0], 0)

# plt.tight_layout()
plt.show()
'''
# %%
from pathlib import Path

path_save = "/home/sachahai/Documents/VAE_manifold/DataSets/4Channel_TIFs/Class_1"
for path in Path('/home/sachahai/Documents/VAE_manifold/DataSets/4Channel_TIFs/Class_1').rglob('*.tif'):
    img_arr = imread(os.path.join(str(path.parent), path.name))
    img_resized = resize(img_arr, (img_arr.shape[0] // 2, img_arr.shape[1] // 2), preserve_range=False, anti_aliasing=False)
    imsave(os.path.join(path_save, path.name), img_as_ubyte(img_resized))
    #print(path.name, " done")