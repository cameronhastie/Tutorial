"""
Matplotlib Animation Example
author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD

Rest is modified by Cameron Hastie.

"""

import numpy as np                  # This library is essential for manipulating data.
import matplotlib.pyplot as plt     # This library is essential for plotting data.
from matplotlib import animation

# test_file = "C:\Users\Cameron\Documents\GitHub\Tutorial\Image_Analysis\data\2022-07-PDMS+TIRF\UnTIRF+BKGD.tif"
test_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/TIRF.tif"
reff_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/unTIRF.tif"
# Grab the test and reference images
test_image = plt.imread(test_file)
test_image = (test_image - np.min(test_image)) / (np.max(test_image) - np.min(test_image))
reff_image = plt.imread(reff_file)
reff_image = (reff_image - np.min(reff_image)) / (np.max(reff_image) - np.min(reff_image))
# Define fraction of image to be cropped
crop_frac = 1/4 
# Define the order of spiral
dirs = ['R', 'U', 'L', 'D']
image_dim = np.size(reff_image[0,:])
low_lim = int(np.round(image_dim*crop_frac))
upp_lim = int(np.round(image_dim*(1-crop_frac)))
shifts = np.floor(np.sqrt(2)*low_lim)
shifts = int(shifts)
diff_images = np.zeros([shifts, upp_lim-low_lim, upp_lim-low_lim])
L2_norm = np.zeros([shifts, 1])

# Initialization function: plot the background of each frame
def init():
    test_movie.set_data(crop_image - crop_image)
    return test_movie,

# animation function.  This is called sequentially
def animate(j):
    low_lim_shift = low_lim - j
    upp_lim_shift = upp_lim - j
    shift_image = test_image[low_lim_shift:upp_lim_shift, low_lim_shift:upp_lim_shift]
    diff_image = crop_image*shift_image
    diff_images[j,:,:] = diff_image
    L2_norm[j] = np.sum(diff_image)
    test_movie.set_data(diff_image)
    ax.set_title(["frame: ", str(j)])
    return test_movie,

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
crop_image = reff_image[low_lim:upp_lim, low_lim:upp_lim]
ax = plt.axes(xlim=(0, upp_lim-low_lim), ylim=(0,upp_lim-low_lim))
test_movie = ax.imshow(crop_image, cmap='gray')
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                            frames=shifts, interval=20, blit=True) 
#anim.save('basic_animation.gif', fps=5)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

plt.show()

plt.plot(L2_norm)
plt.show()