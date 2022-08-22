"""
Matplotlib Animation Example
author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD

Rest is modified by Cameron Hastie.

"""

# LIBRARY IMPORTS ===================================================
#====================================================================
from re import S
from turtle import position
import numpy as np                  # This library is essential for manipulating data.
import matplotlib.pyplot as plt     # This library is essential for plotting data.
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

# CUSTOM FUNCTIONS ==================================================
#====================================================================

def normalize_intensities(image):
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    #test_image = (test_image - np.average(test_image)) / np.average(test_image)
    #reff_image = (reff_image - np.average(reff_image)) / np.average(reff_image)
    return normalized_image

def magnify_image(image, M):
    
    # Define magnifcation of image to simulate by cropping: 
    image_dim = np.size(image[0,:])
    sim_MAG = M # simulating extra 60x magnification.
    beta = image_dim/2
    alph = (1/np.sqrt(sim_MAG))*beta
    low_lim = int(np.floor(beta - alph))
    upp_lim = int(np.ceil(beta + alph))
    image_dim = upp_lim - low_lim

    # Crop images:
    image = image[low_lim:upp_lim, low_lim:upp_lim]
    return image

def compute_G(image):
    Ixy = image
    FFT_Ixy = np.fft.fft2(Ixy) # Fourier ransform the image
    PSD_Ixy = np.fft.ifft2(FFT_Ixy*np.conj(FFT_Ixy)) # Power spectral density
    G_Ixy = np.fft.fftshift(PSD_Ixy) # Autocorrelaton (G_IXY) 
    return FFT_Ixy, PSD_Ixy, G_Ixy

def compute_N(G, lam, NA, pix_per_dl):
    # Compute image dimension 
    image_dim = np.size(G[0,:])
    # TODO: Fit auto_corr to get G0, noise is messing this up.s
    diff_lim = lam/(2*NA) # This is about 2 pixels with our optics
    w02 = np.power(pix_per_dl,2) # Radius of PSF
    image_dim2 = np.power(image_dim, 2)
    G0_est = (np.abs(G[int(image_dim/2), int(image_dim/2)])-np.abs(np.min(G)))
    Num_Beads = image_dim2/(G0_est*np.pi*w02)
    return Num_Beads

def compute_wkernel(image, a):
    image_dim = np.size(image[0,:])
    wave_mat = np.ones([image_dim, image_dim])
    Tab = np.ones([image_dim, image_dim])
    for m in range (image_dim):
        bx = m - image_dim/2
        for k in range (image_dim):
            by = k - image_dim/2
            for i in range(image_dim):
                rx = i - image_dim/2
                for j in range(image_dim):
                    ry = j - image_dim/2
                    del_rba2 = np.power((np.sqrt(np.power(rx-bx,2) + np.power(ry-by,2))/a),2)
                    wave_mat[i,j] = (2 - del_rba2)*np.exp(-1*del_rba2/2) 
            Tab[m,k] = np.sum(np.matmul(image, wave_mat))/a
    return Tab

def compute_WIMS(image, wave_mat, a, b):
    image_dim = np.size(image[0,:])
    wave_mat = compute_wkernel(image_dim, a, b)
    Tab = np.zeros([image_dim, image_dim])
    for i in range(image_dim):
        for j in range(image_dim):
            Tab[i,j] = image[i,j]

def G_heatmaps(image, G):
    # This object is extensive and you can check out the documentation.
    fig, ax = plt.subplots(1,2, figsize=(20, 10), dpi=80)
    ax[0].imshow(image, cmap='jet');
    ax[0].set_xlabel("x pixel", fontsize=15)
    ax[0].set_ylabel("y pixel", fontsize=15)
    ax[1].imshow(np.abs(G), cmap='jet');
    ax[1].set_xlabel("x spatial frequency", fontsize=15)
    ax[1].set_ylabel("y spatial frequency", fontsize=15)
    plt.show()

def G_surfaces(image, G):

#    # TODO: Fix the colorbars to be thinner and less disgusting
    # Compute image dimension 
    image_dim = np.size(image[0,:])
    
    # Make data.
    X = np.arange(-1*image_dim/2, image_dim/2, 1)
    Y = np.arange(-1*image_dim/2, image_dim/2, 1)
    X, Y = np.meshgrid(X, Y)

    alt_axis_nums = np.linspace(-1*image_dim/2, image_dim/2, 5)
    alt_axis_nums = alt_axis_nums.astype(int)
    alt_axis = alt_axis_nums.tolist()
    real_axis = np.linspace(0, np.size(image[0,:]), len(alt_axis))
    #alt_axis = real_axis - np.max(real_axis)/2

    # Plot the surface
    fig, ax = plt.subplots(1,2, subplot_kw={"projection": "3d"})
    surf = ax[0].plot_surface(X, Y, image, cmap=cm.jet,
                        linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    ax[0].set_xlabel("x pixel", fontsize=15)
    ax[0].set_ylabel("y pixel", fontsize=15)
    fig.colorbar(surf, shrink=0.5, orientation="horizontal", pad=0.1, ax=ax[0])
    
    # Plot the surface.
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax[1].plot_surface(X, Y, np.abs(G) - np.abs(np.min(G)), cmap=cm.jet,
                        linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    ax[1].set_xlabel("x spatial shift", fontsize=15)
    ax[1].set_ylabel("y spatial shift", fontsize=15)
    fig.colorbar(surf, shrink=0.5, orientation="horizontal", pad=0.1, ax=ax[1])
    
    plt.show()

def G_profiles(image, G, lam, NA, pixel2difflim):

    image_dim = np.size(image[0,:])
    diff_lim = lam/(2*NA)
    sym_lim = ((image_dim/2)/pixel2difflim)*diff_lim
    asym_lim = (image_dim/pixel2difflim)*diff_lim
    sym_dist = np.linspace(-1*sym_lim, sym_lim, image_dim)
    asym_dist = np.linspace(0, sym_lim, int(image_dim/2))
    fig, ax = plt.subplots(1,2, figsize=(10, 10), dpi=80)
    G_abs = np.abs(G)
    sum_sqrs = np.zeros([int(image_dim/2),1])
    num_elem = np.zeros([int(image_dim/2),1])
    ave_sqrs = np.zeros([int(image_dim/2),1])
    for i in range(int(image_dim/2)):
        top_side = G_abs[i, i:image_dim-i-1]
        bot_side = G_abs[image_dim-i-1, i:image_dim-i-1]
        lft_side = G_abs[i+1:image_dim-i-2, i] # skip one at either end 
        rgt_side = G_abs[i, i+1:image_dim-i-2] # skip one at either end 
        sum_sqrs[i] = np.sum(top_side) + np.sum(bot_side) + +np.sum(lft_side) + np.sum(rgt_side)
        num_elem[i] = np.size(top_side) + np.size(bot_side) + +np.size(lft_side) + np.size(rgt_side)
        ave_sqrs[i] = sum_sqrs[i]/num_elem[i]    
    min_G = np.min(G_abs)
    ax[0].plot(np.log(sym_dist), G_abs[int(image_dim/2), :]-min_G)
    ax[0].plot(np.log(sym_dist), G_abs[:, int(image_dim/2)]-min_G)
    ax[0].plot(np.log(sym_dist), (G_abs[:, int(image_dim/2)] + G_abs[int(image_dim/2), :])/2-min_G)
    ax[0].set_xlabel(r'$\xi$' + " or " + r'$\eta$', fontsize=15)
    ax[0].set_ylabel(r'$G(\xi, 0)$' + " or " + r'$G(0, \eta)$', fontsize=15)
    ax[1].plot(np.log(asym_dist), np.flip(ave_sqrs) - min_G)
    ax[1].set_xlabel(r'$\xi$' + " or " + r'$\eta$', fontsize=15)
    ax[1].set_ylabel("SQUAVE[" + r'$G(\xi, \eta)$]' + "]", fontsize=15)
    plt.show()

# READ IMAGES =======================================================
#====================================================================

data_path = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/"
data_fold = "2022-07-PDMS+TIRF/"
#data_fold = "2022-07-Patterns/"

test_name = "TIRF.tif"
reff_name = "unTIRF.tif"
#reff_name = "Just_Beads_TIRF_7000rpm.tif"
#test_name = "patterns overnight-2.tif"
#reff_name = "rfp-3.tif"

test_file = data_path + data_fold + test_name
reff_file = data_path + data_fold + reff_name

# Grab the test and reference images
test_image = plt.imread(test_file)
reff_image = plt.imread(reff_file)

# NORMALIZE IMAGES ==================================================
#====================================================================

# Normalize (make everything go from 0-1)
test_image = normalize_intensities(test_image)
reff_image = normalize_intensities(reff_image)

# Image dimension, assumes similar sizes
image_dim = np.size(test_image[0,:])

# MAGNIFY IMAGES ====================================================
#====================================================================

magnification = 80 # Nx magnification
test_image = magnify_image(test_image, magnification)
reff_image = magnify_image(reff_image, magnification)

# WAVELET TRANSFORMATION ============================================
#====================================================================

image_dim = np.size(test_image,1)
wave_mats = np.zeros([int(np.sqrt(image_dim)), image_dim, image_dim])
wave_mat = compute_wkernel(test_image, 20)
wave_mat = normalize_intensities(wave_mat)
Tab = test_image*wave_mat
plt.imshow(Tab)
plt.show()

# COMPUTE AUTOCORRELATION ===========================================
#====================================================================

FFT_Ixy_test, PSD_Ixy_test, G_Ixy_test = compute_G(test_image)
FFT_Ixy_reff, PSD_Ixy_reff, G_Ixy_reff = compute_G(reff_image)

# SHOW AUTOCORRELATIONS =============================================
#====================================================================

G_heatmaps(test_image, G_Ixy_test)
G_heatmaps(reff_image, G_Ixy_reff)

# SHOW AUTOCORRELATION SURFACES + CONTOURS ==========================
#====================================================================

G_surfaces(test_image, G_Ixy_test)
G_surfaces(reff_image, G_Ixy_reff)

pixel2difflim = 2
lam = 488e-9
NA = 1.49
G_profiles(test_image, G_Ixy_test, lam, NA, pixel2difflim)
G_profiles(reff_image, G_Ixy_reff, lam, NA, pixel2difflim)

# COMPUTE NUMBER OF PARTICLES IN IMAGE ==============================
#====================================================================

bead_num_TIRF   = str(compute_N(G_Ixy_test, lam, NA, pixel2difflim))
bead_num_unTIRF = str(compute_N(G_Ixy_reff, lam, NA, pixel2difflim))

# Format a print statement
print("TIRF     N = " + bead_num_TIRF + "\n" +
      "unTIRF   N = " + bead_num_unTIRF)

""" WIMS_test = compute_WIMS(test_image)
WIMS_reff = compute_WIMS(reff_image) """