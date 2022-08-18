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

# To get started, let us simply read and display out .tif file.
test_images = [plt.imread("TIRF.tif"), plt.imread("Epi.tif")]
Num_images = len(test_images)
crop_frac = 1/4

# initialization function: plot the background of each frame
def init():
    #test_movie.set_data(np.zeros([image_dim,image_dim]))
    test_movie.set_data(crop_image)
    return test_movie,

# animation function.  This is called sequentially
def animate(j):
    low_lim_shift = low_lim - j
    upp_lim_shift = upp_lim - j
    shift_image = test_image[low_lim_shift:upp_lim_shift, low_lim_shift:upp_lim_shift]
    #L2_norms[j] = np.log(np.sum(np.abs(np.power(crop_image,2)-np.power(shift_image,2)))/np.power(image_dim,2))
    test_movie.set_data(np.abs(crop_image-shift_image))
    return test_movie,

for i in range(1):
    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    test_image = test_images[i]
    image_dim = np.size(test_image[0,:])
    low_lim = int(np.round(image_dim*crop_frac))
    upp_lim = int(np.round(image_dim*(1-crop_frac)))
    crop_image = test_image[low_lim:upp_lim, low_lim:upp_lim]
    ax = plt.axes(xlim=(0, upp_lim-low_lim), ylim=(0,upp_lim-low_lim))
    #test_movie = ax.imshow(np.zeros([image_dim,image_dim]), cmap='gray')
    test_movie = ax.imshow(crop_image, cmap='gray')
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=200, interval=20, blit=True) 
    anim.save('basic_animation.gif', fps=15)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

plt.show()
