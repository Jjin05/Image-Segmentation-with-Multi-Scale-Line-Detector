#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(6/07/2026)s

@author: %(Jordi Jin and Francesc Vázquez Campodarbe, M1)s
"""

# %% Package Imports

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import skimage as sk


# %% MULTI-SCALE DETECTOR FUNCTIONS

# Section II-A.
def I_W_avg(im, W):
    """This function takes an image and a window size W as input.
    It returns a matrix with the dimensions of the image, where each pixel contains the average of the window."""

    filtre_mitjana = (1/(W**2)) * np.ones((W,W))
    I_W_avg_res = sc.ndimage.convolve(im, filtre_mitjana)

    return I_W_avg_res


# Section II-A.
def matrius_linies(W, L, angles=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]):
    """Function that takes as input the window size W and the number of pixels L of a line.
    Returns a series of matrices with lines of L pixels at different angles.
    By default, it works with 12 angles equally spaced by 15 degrees.
    If L is even, it will return matrices with L+1 pixels to ensure symmetry.
    If L is odd, it will work seamlessly."""

    angles = np.array(angles)
    angles = np.radians(angles) # Convert angles to radians.

    # Create the (W,W) matrices x and y with meshgrid.
    x, y = np.meshgrid(np.linspace(-W//2 + 1, W//2, W, endpoint = True), np.linspace(-W//2 + 1, W//2, W, endpoint = True))

    # x_num (y_num) is the exact "x" ("y") value that a point on a straight line should have for a given "y" ("x") and a given angle.
    y_num = np.zeros((W,W,len(angles))) # Create a matrix for each input angle.
    # Add a dimension to "x" (W,W,1) so that it can create a matrix for each different angle.
    y_num = np.expand_dims(x, axis = 2) * np.tan(angles) # Calculate the value of "y" for each angle and different value of x.

    # Do the same for "x".
    x_num = np.zeros((W,W,len(angles)))
    x_num = np.expand_dims(y, axis = 2) /np.where(np.tan(angles) == 0, np.nan, np.tan(angles))

    # It is a (W,W,len(angles)) matrix. There are len(angles) matrices of W by W where each matrix is a line with a different angle.
    matrius_linies_res = np.zeros((W,W,len(angles)))

    # Expand the dimension of "x" and "y" in order to operate with y_num and x_num.
    # We want it to be True at the pixels that lie on the line of a certain angle (each matrix has a different angle).
    # At the same time, we want the number of pixels not to exceed L, which is why we do np.expand_dims(np.abs(x), axis = 2) <= L//2 and np.expand_dims(np.abs(y), axis = 2) <= L//2.

    # Create a mask to prevent the number of pixels from exceeding L.
    C_L = (np.expand_dims(np.abs(x), axis = 2) <= L//2) * (np.expand_dims(np.abs(y), axis = 2) <= L//2)
    # In the first line, for each "x" within the interval [-L//2, L//2], we want to find a "y" that lies on the line of a given angle and set it to True.
    matrius_linies_res = (np.expand_dims(y, axis = 2) == np.round(y_num)) * C_L
    # Perform the same procedure for "y" because for some angles, we cannot find a value of "y" for a given "x", but we can do it for "y".
    # Use the | operator to fill in the missing True values.
    matrius_linies_res = ((np.expand_dims(x, axis = 2) == np.round(x_num)) * C_L) | matrius_linies_res

    return matrius_linies_res


# Section II-A.
def I_W_max(im, W, L):
    """Function that takes as input the image, the window size W, and the number of pixels L of a line.
    Returns a matrix with the dimensions of the image containing the I_W_max values at each pixel."""

    matrius_linies_res = matrius_linies(W, L) # Use 12 angles (by default).
    matrius_I_W = []

    # Iterate over each matrix (12 by default).
    for i in range(matrius_linies_res.shape[2]):
        matriu = sc.ndimage.convolve(im, matrius_linies_res[:,:,i]/np.sum(matrius_linies_res[:,:,i])) 
        # Divided by the number of True values to compute the average.
        
        matrius_I_W.append(matriu)

    # Find the maximum among the 12 matrices.
    I_W_max_res = np.max(np.array(matrius_I_W), axis = 0)

    return I_W_max_res


# Section II-A (Basic Detector if L=W).
# To use it as a basic line detector, the output must be normalized between 0 and 1 and a threshold must be applied.
# It does not do this by default so that the function can be used as part of the multi-scale line detector.
def R_W_L(im, W, L):
    """Function that takes as input the image in floats (values between 0 and 1), the window size W, and the number of pixels L of a line.
    Returns a matrix with the dimensions of the image containing the R_W_L values at each pixel (the response is not necessarily between 0 and 1)."""

    I_W_avg_matrix = I_W_avg(im, W)
    I_W_max_matrix = I_W_max(im, W, L)

    R_W_L_matrix = I_W_max_matrix - I_W_avg_matrix

    return R_W_L_matrix


# Section II-B and II-C (Multi-Scale Detector).
def R_W_combined(im, W):
    """Function that takes as input the image in floats (values between 0 and 1) and the window size W.
    Returns R_W_combined with the dimensions of the image (the response is not necessarily between 0 and 1)."""

    # Create a list to store the linear responses for each value of L.
    R_W_L_list = []

    I_W_avg_matrix = I_W_avg(im, W)

    # Iterate over each value of L.
    for L_i in range(1,W+1,2):
        I_W_max_matrix = I_W_max(im, W, L_i)
        R_W_L_i = I_W_max_matrix - I_W_avg_matrix

        # Standardize the linear response.
        R_W_L_i_mean = np.mean(R_W_L_i)
        R_W_L_i_std = np.std(R_W_L_i)
        R_W_L_i = (R_W_L_i - R_W_L_i_mean)/R_W_L_i_std
      
        # Store the linear response for the current value of L.
        R_W_L_list.append(R_W_L_i)

    # Combine the linear responses.
    n_L = W//2 + 1
    R_W_combined_res = (np.sum(np.array(R_W_L_list), axis = 0) + im)/(n_L + 1) # Set axis = 0 to perform matrix addition.

    return R_W_combined_res


# For retinal images, im is the inverted green channel (as float).
# For satellite images, im is the inverted S channel (as float).
def segmentar(im, W, t):
    """Function that receives the channel to segment as float (values between 0 and 1), the window size W, and a threshold t.
    Segments the image with the multi-scale detector and applies a threshold t.
    Returns the segmented binary image of type uint8."""
    
    # Apply the multi-scale line detector.
    R_W_combined_res = R_W_combined(im, W)
    
    # Binarize with the threshold and convert the result to uint8.
    im_seg = R_W_combined_res > t
    im_seg = (255*im_seg).astype(dtype=np.uint8)
    
    return im_seg


# %% RETINAL IMAGE BORDER HANDLING FUNCTIONS

# Section III-B.
# Creates FOV masks for retinal images.
# The red channel of the image must be provided, as it best separates the retina from the background.
def FOV(im, t=35):
    """Function that receives a retinal image (red channel) in uint8 and a threshold t (default 35).
    Returns the FOV mask of the retinal image, of type bool."""
    
    # Create the mask with the threshold.
    mask = im > t
    # Clean the interior.
    mask = sk.morphology.remove_small_holes(mask)

    return mask


# Section III-B (Alternative Border Handling).
def extension(im, fov, iteracions=40):
    """Function that receives an image (single channel in uint8 or float) and an FOV mask (boolean).
    Returns an image with the retinal border extended using the same data type as the input."""
    
    # Work with copies to avoid modifying the input images.
    ext = im.copy().astype(dtype=np.float64)
    mask = fov.copy()
    
    # Erode the mask to start before the boundary and obtain a smoother transition.
    mask = sk.morphology.erosion(mask, footprint=sk.morphology.disk(3))
    
    # To detect the outer boundary, use a 4-pixel neighborhood.
    kernel_4 = np.array([[0,1,0],
                         [1,1,1],
                         [0,1,0]])
    
    # To compute the average with an 8-pixel neighborhood.
    kernel_8 = np.ones((3,3), dtype=np.float64)
    
    # The extension process is iterative. By default, it performs 40 iterations.
    for i in range(iteracions):
        # Detect the outer boundary and keep only the boundary line.
        mask_dilatat = sk.morphology.dilation(mask, footprint=kernel_4)
        mask_frontera = mask_dilatat != mask
        
        # Apply the FOV mask to reconstruct from its boundary.
        ext_mask = ext*mask
        
        # Apply a filter to compute the average with the 8-pixel neighborhood.
        mitjana = sc.ndimage.convolve(ext_mask, kernel_8, mode='constant', cval=0.0)
        
        # Count how many pixels the window is taking to calculate the average at each pixel.
        # (Count how many pixels of ext_mask belong to the FOV mask).
        num = sc.ndimage.convolve(mask.astype(dtype=np.float64), kernel_8, mode='constant', cval=0.0)
        num[num==0]=1 # To avoid division by 0.
        
        # Adjust the average.
        mitjana = mitjana/num
        
        # Update the extended image and the initial mask.
        ext[mask_frontera] = mitjana[mask_frontera]
        mask[mask_frontera] = True
        
    return ext


# %% POST-PROCESSING FUNCTIONS (Section III-C)

# Directional Morphological Filter.
def morf_dir(im, M, angles=[0,30,60,120,150]):
    """Function that receives a binary image (in bool or uint8) and the size M of the structuring element.
    Applies closings in 5 directions.
    Returns the morphologically modified image (in uint8, between 0 and 255)."""
    
    # Convert the image to bool.
    im_bool = im>0
    
    # Define the structuring elements in the shape of lines at the chosen angles.
    kernel = matrius_linies(M, M, angles)
    # Create an empty image where the results are stored.
    im_post = np.zeros_like(im)
        
    for k in range(len(angles)):
        # Apply closing with the different angles.
        im_closed = sk.morphology.closing(im_bool, footprint=kernel[:,:,k])
        # Store the new pixels with a logical OR.
        im_post = im_post | im_closed
        
    return 255*im_post


# Morphological Reconstruction.
def morf_rec(im, R_rec):
    """Function that takes a binary image (in bool, uint8, or float) and the filter radius R_rec.
    Performs morphological reconstruction, removing noise and reconstructing thin vessels connected to main vessels.
    Returns the reconstructed image (in uint8, between 0 and 255)."""
    
    # Define the initial image as a boolean mask beyond which the reconstruction cannot expand.
    mascara = im > 0
    
    # Create the structuring element.
    kernel = sk.morphology.disk(R_rec)
    # Erode the image, obtaining the marker image to be reconstructed (seed).
    semilla = sk.morphology.erosion(mascara, footprint=kernel)
    
    # Reconstruct from the seed using dilations constrained by the mask.
    im_rec = sk.morphology.reconstruction(seed=semilla, mask=mascara, method="dilation")
    
    return 255*np.uint8(im_rec)


# Post-processing (applies a median filter and the previous functions).
def postprocessat(im, mida_mediana, M, R_rec):
    """Function that receives a binary image (in uint8), the median filter size, the length of 
    the directional morphological filter structuring element, and the radius R_rec of the morphological reconstruction.
    Applies a median filter, a directional morphological filter, and a morphological reconstruction to the image.
    Returns the processed image (in uint8, between 0 and 255)."""
    
    im_post = sc.signal.medfilt(im, kernel_size = mida_mediana)
    im_post = morf_dir(im_post, M)
    im_post = morf_rec(im_post, R_rec)
    
    # The functions already return the image in uint8.

    return im_post


# %% ACCURACY MEASUREMENT FUNCTIONS (Section V-A)

def ACC(im, obs, fov):
    """Takes an image im, a ground truth reference image obs (im and obs must have the same type and dimensions), and an fov mask.
    Compares im and obs by counting the matching pixels within the fov mask.
    Returns the fraction of matching pixels."""

    # Reduce the matrices to be compared to a 1D array with only the values inside the mask.
    im_mask = im[fov]
    obs_mask = obs[fov]

    # Compare the arrays.
    res = np.sum(im_mask==obs_mask) # Matching pixels.
    norm = np.sum(fov==True) # Total pixels compared.
    res_norm = res/norm
    
    return res_norm


def ACC_Local(im, obs, S=3):
    """Takes an image im, a ground truth reference image obs (im and obs must have the same type and dimensions),
    and the size S (default S=3) of a structuring element. 
    Compares im and obs by counting the matching pixels only in a region close to the blood vessels (defined by the dilation of obs by S).
    Returns the fraction of matching pixels in this region."""
    
    structural_element = np.ones((S,S))
    local_mask = sk.morphology.dilation(obs, footprint = structural_element)
    local_mask = local_mask.astype(dtype="bool")

    # Reduce the matrices to be compared to a 1D array with only the values inside the mask.
    im_mask = im[local_mask]
    obs_mask = obs[local_mask]

    # Compare the arrays.
    res = np.sum(im_mask==obs_mask) # Matching pixels.
    norm = np.sum(local_mask==True) # Total pixels compared.
    res_norm = res/norm
    
    return res_norm


# %% STARE DATASET

# We take as observer 1 (ground truth) the one who segmented more vessels.

im_0001 = sk.io.imread("im0001.ppm")
im_0001_obs2 = sk.io.imread("im0001.ah.ppm")
im_0001_obs1 = sk.io.imread("im0001.vk.ppm")

im_0002 = sk.io.imread("im0002.ppm")
im_0002_obs2 = sk.io.imread("im0002.ah.ppm")
im_0002_obs1 = sk.io.imread("im0002.vk.ppm")

im_0003 = sk.io.imread("im0003.ppm")
im_0003_obs2 = sk.io.imread("im0003.ah.ppm")
im_0003_obs1 = sk.io.imread("im0003.vk.ppm")

im_0004 = sk.io.imread("im0004.ppm")
im_0004_obs2 = sk.io.imread("im0004.ah.ppm")
im_0004_obs1 = sk.io.imread("im0004.vk.ppm")

im_0005 = sk.io.imread("im0005.ppm")
im_0005_obs2 = sk.io.imread("im0005.ah.ppm")
im_0005_obs1 = sk.io.imread("im0005.vk.ppm")

im_0044 = sk.io.imread("im0044.ppm")
im_0044_obs2 = sk.io.imread("im0044.ah.ppm")
im_0044_obs1 = sk.io.imread("im0044.vk.ppm")

im_0077 = sk.io.imread("im0077.ppm")
im_0077_obs2 = sk.io.imread("im0077.ah.ppm")
im_0077_obs1 = sk.io.imread("im0077.vk.ppm")

im_0081 = sk.io.imread("im0081.ppm")
im_0081_obs2 = sk.io.imread("im0081.ah.ppm")
im_0081_obs1 = sk.io.imread("im0081.vk.ppm")

im_0082 = sk.io.imread("im0082.ppm")
im_0082_obs2 = sk.io.imread("im0082.ah.ppm")
im_0082_obs1 = sk.io.imread("im0082.vk.ppm")

im_0139 = sk.io.imread("im0139.ppm")
im_0139_obs2 = sk.io.imread("im0139.ah.ppm")
im_0139_obs1 = sk.io.imread("im0139.vk.ppm")

im_0162 = sk.io.imread("im0162.ppm")
im_0162_obs2 = sk.io.imread("im0162.ah.ppm")
im_0162_obs1 = sk.io.imread("im0162.vk.ppm")

im_0163 = sk.io.imread("im0163.ppm")
im_0163_obs2 = sk.io.imread("im0163.ah.ppm")
im_0163_obs1 = sk.io.imread("im0163.vk.ppm")

im_0235 = sk.io.imread("im0235.ppm")
im_0235_obs2 = sk.io.imread("im0235.ah.ppm")
im_0235_obs1 = sk.io.imread("im0235.vk.ppm")

im_0236 = sk.io.imread("im0236.ppm")
im_0236_obs2 = sk.io.imread("im0236.ah.ppm")
im_0236_obs1 = sk.io.imread("im0236.vk.ppm")

im_0239 = sk.io.imread("im0239.ppm")
im_0239_obs2 = sk.io.imread("im0239.ah.ppm")
im_0239_obs1 = sk.io.imread("im0239.vk.ppm")

im_0240 = sk.io.imread("im0240.ppm")
im_0240_obs2 = sk.io.imread("im0240.ah.ppm")
im_0240_obs1 = sk.io.imread("im0240.vk.ppm")

im_0255 = sk.io.imread("im0255.ppm")
im_0255_obs2 = sk.io.imread("im0255.ah.ppm")
im_0255_obs1 = sk.io.imread("im0255.vk.ppm")

im_0291 = sk.io.imread("im0291.ppm")
im_0291_obs2 = sk.io.imread("im0291.ah.ppm")
im_0291_obs1 = sk.io.imread("im0291.vk.ppm")

im_0319 = sk.io.imread("im0319.ppm")
im_0319_obs2 = sk.io.imread("im0319.ah.ppm")
im_0319_obs1 = sk.io.imread("im0319.vk.ppm")

im_0324 = sk.io.imread("im0324.ppm")
im_0324_obs2 = sk.io.imread("im0324.ah.ppm")
im_0324_obs1 = sk.io.imread("im0324.vk.ppm")

# These are 3-channel images (RGB), in uint8 between 0 and 255.
stare = np.array([im_0001, im_0002, im_0003, im_0004, im_0005,
                  im_0044, im_0077, im_0081, im_0082, im_0139,
                  im_0162, im_0163, im_0235, im_0236, im_0239,
                  im_0240, im_0255, im_0291, im_0319, im_0324])


# These are binary images in uint8.
stare_obs1 = np.array([im_0001_obs1, im_0002_obs1, im_0003_obs1, im_0004_obs1, im_0005_obs1,
                       im_0044_obs1, im_0077_obs1, im_0081_obs1, im_0082_obs1, im_0139_obs1,
                       im_0162_obs1, im_0163_obs1, im_0235_obs1, im_0236_obs1, im_0239_obs1,
                       im_0240_obs1, im_0255_obs1, im_0291_obs1, im_0319_obs1, im_0324_obs1])

# These are binary images in uint8.
stare_obs2 = np.array([im_0001_obs2, im_0002_obs2, im_0003_obs2, im_0004_obs2, im_0005_obs2,
                       im_0044_obs2, im_0077_obs2, im_0081_obs2, im_0082_obs2, im_0139_obs2,
                       im_0162_obs2, im_0163_obs2, im_0235_obs2, im_0236_obs2, im_0239_obs2,
                       im_0240_obs2, im_0255_obs2, im_0291_obs2, im_0319_obs2, im_0324_obs2])


# %% # STARE Masks.
# Create the FOV masks for this dataset using the red channel of the retinal images.
# The masks are boolean.

stare_mask = []
for im in stare:
    stare_mask.append(FOV(im[:,:,0]))
stare_mask = np.array(stare_mask)

# Erode the FOV masks to avoid taking the border of the segmented image.
structural_element = np.ones((15,15))
stare_mask_erosion = []
for mask in stare_mask:
    stare_mask_erosion.append(sk.morphology.erosion(mask, footprint=structural_element))
stare_mask_erosion = np.array(stare_mask_erosion)


# %% Segment STARE images.

W = 15
t = 0.56

stare_seg = []
for im in stare[:,:,:,1]/255: # Take the green channel as float. 
    stare_seg.append(segmentar(1-im, W, t)) # Invert the green channel to segment it.
stare_seg = np.array(stare_seg)

# stare_seg is a set of 20 uint8 binary images.


# %% Check by visual inspection that the FOV masks created for STARE work properly.

for i in range(20):
    plt.figure()
    fig, axes = plt.subplots(1, 3, layout='constrained', figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(stare_seg[i], cmap="gray")
    plt.title("Imatge Segmentada")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(stare_seg[i]*stare_mask[i], cmap="gray")
    plt.title("Aplicant Màscara FOV")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(stare_seg[i]*stare_mask_erosion[i], cmap="gray")
    plt.title("Aplicant Màscara FOV Erosionada")
    plt.axis("off")

    plt.show()

    
# %% DRIVE DATASET

# To read some of the following images, it may be necessary to install the following package (running in terminal):
# pip install imagecodecs

im_21 = sk.io.imread("21_training.tif")
im_21_obs = sk.io.imread("21_manual1.gif")[0]
im_21_mask = sk.io.imread("21_training_mask.gif")[0]

im_22 = sk.io.imread("22_training.tif")
im_22_obs = sk.io.imread("22_manual1.gif")[0]
im_22_mask = sk.io.imread("22_training_mask.gif")[0]

im_23 = sk.io.imread("23_training.tif")
im_23_obs = sk.io.imread("23_manual1.gif")[0]
im_23_mask = sk.io.imread("23_training_mask.gif")[0]

im_24 = sk.io.imread("24_training.tif")
im_24_obs = sk.io.imread("24_manual1.gif")[0]
im_24_mask = sk.io.imread("24_training_mask.gif")[0]

im_25 = sk.io.imread("25_training.tif")
im_25_obs = sk.io.imread("25_manual1.gif")[0]
im_25_mask = sk.io.imread("25_training_mask.gif")[0]

im_26 = sk.io.imread("26_training.tif")
im_26_obs = sk.io.imread("26_manual1.gif")[0]
im_26_mask = sk.io.imread("26_training_mask.gif")[0]

im_27 = sk.io.imread("27_training.tif")
im_27_obs = sk.io.imread("27_manual1.gif")[0]
im_27_mask = sk.io.imread("27_training_mask.gif")[0]

im_28 = sk.io.imread("28_training.tif")
im_28_obs = sk.io.imread("28_manual1.gif")[0]
im_28_mask = sk.io.imread("28_training_mask.gif")[0]

im_29 = sk.io.imread("29_training.tif")
im_29_obs = sk.io.imread("29_manual1.gif")[0]
im_29_mask = sk.io.imread("29_training_mask.gif")[0]

im_30 = sk.io.imread("30_training.tif")
im_30_obs = sk.io.imread("30_manual1.gif")[0]
im_30_mask = sk.io.imread("30_training_mask.gif")[0]

im_31 = sk.io.imread("31_training.tif")
im_31_obs = sk.io.imread("31_manual1.gif")[0]
im_31_mask = sk.io.imread("31_training_mask.gif")[0]

im_32 = sk.io.imread("32_training.tif")
im_32_obs = sk.io.imread("32_manual1.gif")[0]
im_32_mask = sk.io.imread("32_training_mask.gif")[0]

im_33 = sk.io.imread("33_training.tif")
im_33_obs = sk.io.imread("33_manual1.gif")[0]
im_33_mask = sk.io.imread("33_training_mask.gif")[0]

im_34 = sk.io.imread("34_training.tif")
im_34_obs = sk.io.imread("34_manual1.gif")[0]
im_34_mask = sk.io.imread("34_training_mask.gif")[0]

im_35 = sk.io.imread("35_training.tif")
im_35_obs = sk.io.imread("35_manual1.gif")[0]
im_35_mask = sk.io.imread("35_training_mask.gif")[0]

im_36 = sk.io.imread("36_training.tif")
im_36_obs = sk.io.imread("36_manual1.gif")[0]
im_36_mask = sk.io.imread("36_training_mask.gif")[0]

im_37 = sk.io.imread("37_training.tif")
im_37_obs = sk.io.imread("37_manual1.gif")[0]
im_37_mask = sk.io.imread("37_training_mask.gif")[0]

im_38 = sk.io.imread("38_training.tif")
im_38_obs = sk.io.imread("38_manual1.gif")[0]
im_38_mask = sk.io.imread("38_training_mask.gif")[0]

im_39 = sk.io.imread("39_training.tif")
im_39_obs = sk.io.imread("39_manual1.gif")[0]
im_39_mask = sk.io.imread("39_training_mask.gif")[0]

im_40 = sk.io.imread("40_training.tif")
im_40_obs = sk.io.imread("40_manual1.gif")[0]
im_40_mask = sk.io.imread("40_training_mask.gif")[0]

# These are 3-channel images (RGB), in uint8 between 0 and 255.
drive = np.array([im_21, im_22, im_23, im_24, im_25,
                  im_26, im_27, im_28, im_29, im_30,
                  im_31, im_32, im_33, im_34, im_35,
                  im_36, im_37, im_38, im_39, im_40])

# These are binary images in uint8.
drive_obs = np.array([im_21_obs, im_22_obs, im_23_obs, im_24_obs, im_25_obs,
                      im_26_obs, im_27_obs, im_28_obs, im_29_obs, im_30_obs,
                      im_31_obs, im_32_obs, im_33_obs, im_34_obs, im_35_obs,
                      im_36_obs, im_37_obs, im_38_obs, im_39_obs, im_40_obs])

# These are binary images in uint8.
drive_mask = np.array([im_21_mask, im_22_mask, im_23_mask, im_24_mask, im_25_mask,
                       im_26_mask, im_27_mask, im_28_mask, im_29_mask, im_30_mask,
                       im_31_mask, im_32_mask, im_33_mask, im_34_mask, im_35_mask,
                       im_36_mask, im_37_mask, im_38_mask, im_39_mask, im_40_mask])

# Make the FOV masks boolean.
drive_mask = drive_mask.astype(dtype="bool")

# Create the eroded FOV masks. 
structural_element = np.ones((15,15))
drive_mask_erosion=[]
for mask in drive_mask:
    drive_mask_erosion.append(sk.morphology.erosion(mask, footprint=structural_element))
drive_mask_erosion = np.array(drive_mask_erosion)


# %% Segment DRIVE images.

W = 15
t = 0.56

drive_seg = []
for im in drive[:,:,:,1]/255: # Take the green channel as float. 
    drive_seg.append(segmentar(1-im, W, t)) # Invert the green channel to segment it.
drive_seg = np.array(drive_seg)

# drive_seg is a set of 20 uint8 binary images.


# %% Check by visual inspection that the extended FOV masks for DRIVE work properly and do not remove information.

for i in range(20):
    plt.figure()
    fig, axes = plt.subplots(1, 3, layout='constrained', figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(drive_seg[i], cmap="gray")
    plt.title("Imatge Segmentada")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(drive_seg[i]*drive_mask[i], cmap="gray")
    plt.title("Aplicant Màscara FOV")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(drive_seg[i]*drive_mask_erosion[i], cmap="gray")
    plt.title("Aplicant Màscara FOV Erosionada")
    plt.axis("off")

    plt.show()


# %% ALTERNATIVE BORDER HANDLING FOR DRIVE AND STARE RETINAL IMAGES

# Extend the borders of the green channel and segment the images (use the uneroded mask for the function).

stare_vores = []
for i in range(20):
    stare_vores.append(extension(stare[i,:,:,1], stare_mask[i]))
stare_vores = np.array(stare_vores)

drive_vores = []
for i in range(20):
    drive_vores.append(extension(drive[i,:,:,1], drive_mask[i]))
drive_vores = np.array(drive_vores)

W = 15
t = 0.56

stare_vores_seg = []
for im in stare_vores:
    stare_vores_seg.append(segmentar(1-im/255, W, t)) # Convert the green channel to float and invert it before segmenting.
stare_vores_seg = np.array(stare_vores_seg) # 20 uint8 binary images.

drive_vores_seg = []
for im in drive_vores:
    drive_vores_seg.append(segmentar(1-im/255, W, t)) # Convert the green channel to float and invert it before segmenting.
drive_vores_seg = np.array(drive_vores_seg) # 20 uint8 binary images.


# %% POST-PROCESSING OF STARE AND DRIVE RETINAL IMAGES

mida_mediana = 3
M = 5
R_rec = 1

# Without border handling.
stare_seg_post=[]
for im in stare_seg:
    stare_seg_post.append( postprocessat(im, mida_mediana, M, R_rec) )
stare_seg_post = np.array(stare_seg_post) # 20 uint8 binary images.

drive_seg_post=[]
for im in drive_seg:
    drive_seg_post.append( postprocessat(im, mida_mediana, M, R_rec) )
drive_seg_post = np.array(drive_seg_post) # 20 uint8 binary images.


# With border handling.
stare_vores_seg_post=[]
for im in stare_vores_seg:
    stare_vores_seg_post.append( postprocessat(im, mida_mediana, M, R_rec) )
stare_vores_seg_post = np.array(stare_vores_seg_post) # 20 uint8 binary images.

drive_vores_seg_post=[]
for im in drive_vores_seg:
    drive_vores_seg_post.append( postprocessat(im, mida_mediana, M, R_rec) )
drive_vores_seg_post = np.array(drive_vores_seg_post) # 20 uint8 binary images.


# %% ACCURACY METRICS
# %% ACC

# Compare the segmented DRIVE images (applying the database masks) with those segmented by the observer.
# Apply the eroded FOV mask.
drive_acc = []
for i in range(20):
    im = drive_seg[i]*drive_mask_erosion[i]
    obs = drive_obs[i]
    fov = drive_mask_erosion[i]
    drive_acc.append(ACC(im, obs, fov))
drive_acc = np.array(drive_acc)
drive_acc_avg = drive_acc.mean()

# Same with the segmented and post-processed images.
# Apply the eroded FOV mask.
drive_acc_post = []
for i in range(20):
    im = drive_seg_post[i]*drive_mask_erosion[i]
    obs = drive_obs[i]
    fov = drive_mask_erosion[i]
    drive_acc_post.append(ACC(im, obs, fov))
drive_acc_post = np.array(drive_acc_post)
drive_acc_post_avg = drive_acc_post.mean()

# Same with the segmented images with border handling and without post-processing.
# Apply the uneroded FOV mask.
drive_acc_vores = []
for i in range(20):
    im = drive_vores_seg[i]*drive_mask[i]
    obs = drive_obs[i]
    fov = drive_mask[i]
    drive_acc_vores.append(ACC(im, obs, fov))
drive_acc_vores = np.array(drive_acc_vores)
drive_acc_vores_avg = drive_acc_vores.mean()

# Same with the segmented images with border handling and post-processing.
# Apply the uneroded FOV mask.
drive_acc_vores_post = []
for i in range(20):
    im = drive_vores_seg_post[i]*drive_mask[i]
    obs = drive_obs[i]
    fov = drive_mask[i]
    drive_acc_vores_post.append(ACC(im, obs, fov))
drive_acc_vores_post = np.array(drive_acc_vores_post)
drive_acc_vores_post_avg = drive_acc_vores_post.mean()


# Compare the segmented STARE images (applying the created masks) with those segmented by observer 1.
# Apply the eroded FOV mask.
stare_acc = []
for i in range(20):
    im = stare_seg[i]*stare_mask_erosion[i]
    obs = stare_obs1[i]
    fov = stare_mask_erosion[i]
    stare_acc.append(ACC(im, obs, fov))
stare_acc = np.array(stare_acc)
stare_acc_avg = stare_acc.mean()

# Same with the segmented and post-processed images.
# Apply the eroded FOV mask.
stare_acc_post = []
for i in range(20):
    im = stare_seg_post[i]*stare_mask_erosion[i]
    obs = stare_obs1[i]
    fov = stare_mask_erosion[i]
    stare_acc_post.append(ACC(im, obs, fov))
stare_acc_post = np.array(stare_acc_post)
stare_acc_post_avg = stare_acc_post.mean()

# Same with the segmented images with border handling and without post-processing.
# Apply the uneroded FOV mask.
stare_acc_vores = []
for i in range(20):
    im = stare_vores_seg[i]*stare_mask[i]
    obs = stare_obs1[i]
    fov = stare_mask[i]
    stare_acc_vores.append(ACC(im, obs, fov))
stare_acc_vores = np.array(stare_acc_vores)
stare_acc_vores_avg = stare_acc_vores.mean()

# Same with the segmented images with border handling and post-processing.
# Apply the uneroded FOV mask.
stare_acc_vores_post = []
for i in range(20):
    im = stare_vores_seg_post[i]*stare_mask[i]
    obs = stare_obs1[i]
    fov = stare_mask[i]
    stare_acc_vores_post.append(ACC(im, obs, fov))
stare_acc_vores_post = np.array(stare_acc_vores_post)
stare_acc_vores_post_avg = stare_acc_vores_post.mean()

# Compare the STARE dataset images segmented by observer 2 with those segmented by observer 1.
# Apply the uneroded FOV mask.
stare_acc_12 = []
for i in range(20):
    im = stare_obs2[i]
    obs = stare_obs1[i]
    fov = stare_mask[i]
    stare_acc_12.append(ACC(im, obs, fov))
stare_acc_12 = np.array(stare_acc_12)
stare_acc_avg_12 = stare_acc_12.mean()


# %% Local ACC

# Compare the segmented DRIVE images (applying the database masks) with those segmented by the observer.
# Apply the eroded FOV mask.
drive_acc_Local = []
for i in range(20):
    im = drive_seg[i]*drive_mask_erosion[i]
    obs = drive_obs[i]
    drive_acc_Local.append(ACC_Local(im, obs))
drive_acc_Local = np.array(drive_acc_Local)
drive_acc_Local_avg = drive_acc_Local.mean()

# Same with the segmented and post-processed images.
# Apply the eroded FOV mask.
drive_acc_Local_post = []
for i in range(20):
    im = drive_seg_post[i]*drive_mask_erosion[i]
    obs = drive_obs[i]
    drive_acc_Local_post.append(ACC_Local(im, obs))
drive_acc_Local_post = np.array(drive_acc_Local_post)
drive_acc_Local_post_avg = drive_acc_Local_post.mean()

# Same with the segmented images with border handling and without post-processing.
# Apply the uneroded FOV mask.
drive_acc_Local_vores = []
for i in range(20):
    im = drive_vores_seg[i]*drive_mask[i]
    obs = drive_obs[i]
    drive_acc_Local_vores.append(ACC_Local(im, obs))
drive_acc_Local_vores = np.array(drive_acc_Local_vores)
drive_acc_Local_vores_avg = drive_acc_Local_vores.mean()

# Same with the segmented images with border handling and post-processing.
# Apply the uneroded FOV mask.
drive_acc_Local_vores_post = []
for i in range(20):
    im = drive_vores_seg_post[i]*drive_mask[i]
    obs = drive_obs[i]
    drive_acc_Local_vores_post.append(ACC_Local(im, obs))
drive_acc_Local_vores_post = np.array(drive_acc_Local_vores_post)
drive_acc_Local_vores_post_avg = drive_acc_Local_vores_post.mean()


# Compare the segmented STARE images (applying the created masks) with those segmented by observer 1.
# Apply the eroded FOV mask.
stare_acc_Local = []
for i in range(20):
    im = stare_seg[i]*stare_mask_erosion[i]
    obs = stare_obs1[i]
    stare_acc_Local.append(ACC_Local(im, obs))
stare_acc_Local = np.array(stare_acc_Local)
stare_acc_Local_avg = stare_acc_Local.mean()

# Same with the segmented and post-processed images.
# Apply the eroded FOV mask.
stare_acc_Local_post = []
for i in range(20):
    im = stare_seg_post[i]*stare_mask_erosion[i]
    obs = stare_obs1[i]
    stare_acc_Local_post.append(ACC_Local(im, obs))
stare_acc_Local_post = np.array(stare_acc_Local_post)
stare_acc_Local_post_avg = stare_acc_Local_post.mean()

# Same with the segmented images with border handling and without post-processing.
# Apply the uneroded FOV mask.
stare_acc_Local_vores = []
for i in range(20):
    im = stare_vores_seg[i]*stare_mask[i]
    obs = stare_obs1[i]
    stare_acc_Local_vores.append(ACC_Local(im, obs))
stare_acc_Local_vores = np.array(stare_acc_Local_vores)
stare_acc_Local_vores_avg = stare_acc_Local_vores.mean()

# Same with the segmented images with border handling and post-processing.
# Apply the uneroded FOV mask.
stare_acc_Local_vores_post = []
for i in range(20):
    im = stare_vores_seg_post[i]*stare_mask[i]
    obs = stare_obs1[i]
    stare_acc_Local_vores_post.append(ACC_Local(im, obs))
stare_acc_Local_vores_post = np.array(stare_acc_Local_vores_post)
stare_acc_Local_vores_post_avg = stare_acc_Local_vores_post.mean()

# Compare the STARE dataset images segmented by observer 2 with those segmented by observer 1.
stare_acc_Local_12 = []
for i in range(20):
    im = stare_obs2[i]
    obs = stare_obs1[i]
    stare_acc_Local_12.append(ACC_Local(im, obs))
stare_acc_Local_12 = np.array(stare_acc_Local_12)
stare_acc_Local_avg_12 = stare_acc_Local_12.mean()


# %% SUPPORT RETINAL IMAGES FOR THE PAPER

# Load a retinal image.
img = sk.data.retina()


# Create an FOV mask (with the red channel) and the eroded version.
fov = FOV(img[:,:,0])
structural_element = np.ones((15,15))
fov_erosion = sk.morphology.erosion(fov, footprint=structural_element)


# Create the extended version of the channel to segment (green).
img_ext = extension(img[:,:,1], fov)


# Segment with the basic detector.
W = 15
L = 15
t = 0.25

img_g_inv = 1 - img[:,:,1]/255
R_W_L_matrix = R_W_L(img_g_inv, W, L)
R_W_L_escalat = (R_W_L_matrix - R_W_L_matrix.min())/(R_W_L_matrix.max() - R_W_L_matrix.min())
seg_basic = R_W_L_escalat > t


# Segment with the multi-scale detector.
W = 15
t = 0.56

img_g_inv = 1 - img[:,:,1]/255
img_ext_g_inv = 1 - img_ext/255
seg = segmentar(img_g_inv, W, t)
seg_ext = segmentar(img_ext_g_inv, W, t)


# Apply post-processing to the images segmented with the multi-scale detector.
mida_mediana = 3
M = 5
R_rec = 2

seg_post = postprocessat(seg, mida_mediana, M, R_rec)
seg_ext_post = postprocessat(seg_ext, mida_mediana, M, R_rec)


# %% Plot: Basic Detector vs Multi-Scale Detector.

plt.figure()
fig, axes = plt.subplots(1, 2, layout='constrained', figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(seg_basic, cmap="gray")
plt.title("a", fontsize = 20, loc="left")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg, cmap="gray")
plt.title("b", fontsize = 20, loc="left")
plt.axis("off")

plt.show()

# %% Plot: Border Handling Comparison (Initial Channel and Segmentation With and Without FOV).

plt.figure()
fig, axes = plt.subplots(2, 3, layout='constrained', figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(img[:,:,1], cmap="gray")
plt.title("a", fontsize = 25, loc="left")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(img_ext, cmap="gray")
plt.title("d", fontsize = 25, loc="left")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(seg, cmap="gray")
plt.title("b", fontsize = 25, loc="left")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(seg_ext, cmap="gray")
plt.title("e", fontsize = 25, loc="left")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(seg*fov_erosion, cmap="gray")
plt.title("c", fontsize = 25, loc="left")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(seg_ext*fov, cmap="gray")
plt.title("f", fontsize = 25, loc="left")
plt.axis("off")

plt.show()

# %% Plot: Segmentation Comparison Without and With Post-Processing.

plt.figure()
fig, axes = plt.subplots(1, 2, layout='constrained', figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(seg, cmap="gray")
plt.title("a", fontsize = 20, loc="left")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg_post, cmap="gray")
plt.title("b", fontsize = 20, loc="left")
plt.axis("off")

plt.show()

# %% Plot: Comparison of Image Segmented by Observer 1 and Segmentation with the Proposed Method and Border Handling Without Post-Processing.

# Use image 0255 from STARE, for which we obtained the highest Local ACC.
plt.figure()
fig, axes = plt.subplots(1, 3, layout='constrained', figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(stare_obs1[10], cmap="gray")
plt.title("a", fontsize = 25, loc="left")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(stare_obs2[10], cmap="gray")
plt.title("b", fontsize = 25, loc="left")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(stare_vores_seg[10]*stare_mask[10], cmap="gray")
plt.title("c", fontsize = 25, loc="left")
plt.axis("off")

plt.show()


# %% APPLICATION TO SATELLITE IMAGERY
# %% FIRST IMAGE

# Import the ground truth segmented image and the original image to segment.
# Images in uint8.
im1_referencia = sk.io.imread("11128870_15.tif")
im1 = sk.io.imread("11128870_15.tiff")

# Reduce the resolution of the image to segment to work with acceptable execution times.
im1 = im1[::3,::3,:]
# Convert the image to HSV color space.
im1_hsv = sk.color.rgb2hsv(im1)
# Take the inverted S channel (the one we want to segment).
im1_s_inv = 1 - im1_hsv[:,:,1]/255

# Segmentation parameters.
W = 51
t = 0.5
im1_seg = segmentar(im1_s_inv, W, t) # The function inverts the channel internally.

# Post-processing parameters.
mida_mediana = 1
M = 2
R_rec = 5
im1_post = postprocessat(im1_seg, mida_mediana, M, R_rec)

# %% Plot.

plt.figure()
fig, axes = plt.subplots(1, 4, layout='constrained', figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(im1)
plt.title("a", fontsize=30, loc="left")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(im1_referencia, cmap="gray")
plt.title("b", fontsize=30, loc="left")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(im1_seg, cmap="gray")
plt.title("c", fontsize=30, loc="left")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(im1_post, cmap="gray")
plt.title("d", fontsize=30, loc="left")
plt.axis("off")

plt.show()


# %% SECOND IMAGE

# Import the ground truth segmented image and the original image to segment.
# Images in uint8.
im2_referencia = sk.io.imread("17878780_15.tif")
im2 = sk.io.imread("17878780_15.tiff")

# Reduce the resolution of the image to segment to work with acceptable execution times.
im2 = im2[::3,::3,:]
# Convert the image to HSV color space.
im2_hsv = sk.color.rgb2hsv(im2)
# Take the inverted S channel (the one we want to segment).
im2_s_inv = 1 - im2_hsv[:,:,1]/255

# Segmentation parameters.
W = 31
t = 0.5
im2_seg = segmentar(im2_s_inv, W, t) # The function inverts the channel internally.

# Post-processing parameters.
mida_mediana = 5
M = 5
R_rec = 10
im2_post = postprocessat(im2_seg, mida_mediana, M, R_rec)

# %% Plot.

plt.figure()
fig, axes = plt.subplots(1, 4, layout='constrained', figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(im2)
plt.title("a", fontsize=30, loc="left")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(im2_referencia, cmap="gray")
plt.title("b", fontsize=30, loc="left")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(im2_seg, cmap="gray")
plt.title("c", fontsize=30, loc="left")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(im2_post, cmap="gray")
plt.title("d", fontsize=30, loc="left")
plt.axis("off")

plt.show()


# %% THIRD IMAGE

# Import the ground truth segmented image and the original image to segment.
# Images in uint8.
im3_referencia = sk.io.imread("18178780_15.tif")
im3 = sk.io.imread("18178780_15.tiff")

# Reduce the resolution of the image to segment to work with acceptable execution times.
im3 = im3[::3,::3,:]
# Convert the image to HSV color space.
im3_hsv = sk.color.rgb2hsv(im3)
# Take the inverted S channel (the one we want to segment).
im3_s_inv = 1 - im3_hsv[:,:,1]/255

# Segmentation parameters.
W = 31
t = 0.5
im3_seg = segmentar(im3_s_inv, W, t) # The function inverts the channel internally.

# Post-processing parameters.
mida_mediana = 5
M = 3
R_rec = 8
im3_post = postprocessat(im3_seg, mida_mediana, M, R_rec)

# %% Plot.

plt.figure()
fig, axes = plt.subplots(1, 4, layout='constrained', figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(im3)
plt.title("a", fontsize=30, loc="left")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(im3_referencia, cmap="gray")
plt.title("b", fontsize=30, loc="left")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(im3_seg, cmap="gray")
plt.title("c", fontsize=30, loc="left")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(im3_post, cmap="gray")
plt.title("d", fontsize=30, loc="left")
plt.axis("off")

plt.show()