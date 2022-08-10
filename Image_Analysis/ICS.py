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
import numpy as np                  # This library is essential for manipulating data.
import matplotlib.pyplot as plt     # This library is essential for plotting data.
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

# CUSTOM FUNCTIONS ==================================================
#====================================================================

# INIT IMAGES =======================================================
#====================================================================

#test_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/TIRF+BKGD.tif"
reff_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/TIRF+BKGD.tif"
test_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/unTIRF.tif"
#reff_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/unTIRF.tif"

# Grab the test and reference images
test_image = plt.imread(test_file)
reff_image = plt.imread(reff_file)

# Normalize (make everything go from 0-1)
test_image = (test_image - np.min(test_image)) / (np.max(test_image) - np.min(test_image))
reff_image = (reff_image - np.min(reff_image)) / (np.max(reff_image) - np.min(reff_image))

# Normalize (make everything relative to the average)
#test_image = (test_image - np.average(test_image)) / np.average(test_image)
#reff_image = (reff_image - np.average(reff_image)) / np.average(reff_image)

# Image dimension, assumes similar sizes
image_dim = np.size(test_image[0,:])

# INIT SHIFTS =======================================================
#====================================================================

# Define fraction of image boarder to be cropped: 
crop_frac = 0
low_lim = int(np.floor(image_dim*crop_frac))
upp_lim = int(np.floor(image_dim*(1-crop_frac)))
image_dim = upp_lim - low_lim

# Crop images:
test_image = test_image[low_lim:upp_lim, low_lim:upp_lim]
reff_image = reff_image[low_lim:upp_lim, low_lim:upp_lim]
# Max number of shifts
num_shifts = int(low_lim/10)
num_angles = num_shifts

FFT_Ixy = np.fft.fft2(test_image)
G_en = np.fft.fftshift(np.fft.ifft2(FFT_Ixy*np.conj(FFT_Ixy)))

# Again, we can construct a figure using similar parameters.
fig, ax = plt.subplots(1,2, figsize=(20, 10), dpi=80)
alt_axis_nums = np.linspace(-1*image_dim/2, image_dim/2, 5)
alt_axis_nums = alt_axis_nums.astype(int)
alt_axis = alt_axis_nums.tolist()
real_axis = np.linspace(0, np.size(test_image[0,:]), len(alt_axis))
alt_axis = real_axis - np.max(real_axis)/2
# We take the log to make the transformed image visible with linear contrast.
ax[0].imshow(test_image, cmap=cm.YlGnBu)
ax[0].set_xlabel("x pixel", fontsize=15)
ax[0].set_xticks(real_axis)
ax[0].set_xticklabels(alt_axis)
ax[0].set_ylabel("y pixel", fontsize=15)
ax[0].set_yticks(real_axis)
ax[0].set_yticklabels(np.flip(alt_axis))

# We take the log to make the transformed image visible with linear contrast.
crop_frac = 1/2.5
low_lim = int(np.floor(image_dim*crop_frac))
upp_lim = int(np.floor(image_dim*(1-crop_frac)))
image_dim_c = upp_lim - low_lim

# Again, we can construct a figure using similar parameters.
alt_axis_nums = np.linspace(-1*image_dim_c/2, image_dim_c/2, 5)
alt_axis_nums = alt_axis_nums.astype(int)
alt_axis = alt_axis_nums.tolist()
real_axis = np.linspace(low_lim, image_dim_c, len(alt_axis))
alt_axis = real_axis - np.max(real_axis)/2
ax[1].imshow(np.abs(G_en[low_lim:upp_lim, low_lim:upp_lim]), cmap=cm.YlGnBu)
ax[1].set_xlabel("x spatial shift", fontsize=15)
ax[1].set_xticks(real_axis)
ax[1].set_xticklabels(alt_axis)
ax[1].set_ylabel("y spatial shift", fontsize=15)
ax[1].set_yticks(real_axis)
ax[1].set_yticklabels(np.flip(alt_axis))
plt.show()

# Make data.
X = np.arange(-1*image_dim/2, image_dim/2, 1)
Y = np.arange(-1*image_dim/2, image_dim/2, 1)
X, Y = np.meshgrid(X, Y)

# Plot the surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, np.abs(G_en), cmap=cm.jet,
                       linewidth=0, antialiased=False)
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

# Plot the surface.
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, test_image, cmap=cm.jet,
                       linewidth=0, antialiased=False)
#plt.show()

NA = 1.49
lam = 488e-9
diff_lim = lam/(2*NA) # This is about 2 pixels with our optics
pix_per_dl = 2
image_dim2 = np.power((image_dim/pix_per_dl)*diff_lim, 2)
w02 = np.power(diff_lim,2)
Num_Beads = image_dim2/(np.abs(G_en[0,0])*np.pi*w02)
print(Num_Beads)

fig = plt.figure(num=None, figsize=(8, 6), dpi=80)
G_en_abs = np.abs(G_en)
sum_sqrs = np.zeros([int(image_dim/2),1])
num_elem = np.zeros([int(image_dim/2),1])
ave_sqrs = np.zeros([int(image_dim/2),1])
for i in range(int(image_dim/2)):
    top_side = G_en_abs[i, i:image_dim-i]
    bot_side = G_en_abs[image_dim-i, i:image_dim-i]
    lft_side = G_en_abs[i+1:image_dim-i-1, i] # skip one at either end 
    rgt_side = G_en_abs[i, i+1:image_dim-i-1] # skip one at either end 
    sum_sqrs[i] = np.sum(top_side) + np.sum(bot_side) + +np.sum(lft_side) + np.sum(rgt_side)
    num_elem[i] = np.size(top_side) + np.size(bot_side) + +np.size(lft_side) + np.size(rgt_side)
    ave_sqrs[i] = sum_sqrs[i]/num_elem[i]    
    plt.plot(top_side)
    plt.plot(bot_side)
    plt.plot(lft_side)
    plt.plot(rgt_side)
    plt.show()
plt.plot(ave_sqrs)
plt.show()

""" # Init transformation matrix ~ movie
G_en = np.zeros([upp_lim-low_lim, upp_lim-low_lim])
for i in range(num_shifts):
    for j in range (num_shifts):
        xi = i
        eta = j
        ave = 0
        for m in range(num_shifts-i):
            for n in range(num_shifts-j): """


