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

#test_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/UnTIRF+BKGD.tif"
#reff_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/TIRF+BKGD.tif"
test_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/TIRF.tif"
reff_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/TIRF.tif"
# Grab the test and reference images
test_image = plt.imread(test_file)
test_image = np.log(test_image)
test_image = (test_image - np.min(test_image)) / (np.max(test_image) - np.min(test_image))
reff_image = plt.imread(reff_file)
reff_image = np.log(reff_image)
reff_image = (reff_image - np.min(reff_image)) / (np.max(reff_image) - np.min(reff_image))
# Define fraction of image to be cropped
crop_frac = 1/10
# Define the order of spiral
directions = ['R', 'U', 'L', 'D']
image_dim = np.size(reff_image[0,:])
low_lim = int(np.floor(image_dim*crop_frac))
upp_lim = int(np.floor(image_dim*(1-crop_frac)))
shifts = int(low_lim)
tran_images = np.zeros([shifts, upp_lim-low_lim, upp_lim-low_lim])
direction_reps = np.zeros([2*shifts,1])
for i in range(shifts):
    loop_shifted_ind = 2*i
    direction_reps[loop_shifted_ind] = i+1
    direction_reps[loop_shifted_ind+1] = i+1
L2_norm = np.zeros([shifts, 1])

dir_counter = 0
step_counter = 0

# Initialization function: plot the background of each frame
def init():
    test_movie.set_data(crop_image - crop_image)
    return test_movie,

# animation function.  This is called sequentially
def animate(j):
    # directions = [R,U,L,D]    |    direction_repeats = [1,1,3,3,4,4,5,5,...,Nshift,Nshift]
    direct = directions[np.mod(j,len(directions)-1)]
    shift_image = shift_crop_dir(direct,j)
    tran_image = crop_image*shift_image
    L2_norm[j] = np.sum(crop_image*shift_image)
    #L2_norm[j] = np.sum(crop_image*shift_image)/np.sum(crop_image*crop_image)
    #tran_image = np.power((crop_image-shift_image),2)
    #L2_norm[j] = np.sqrt(np.sum(np.power((crop_image-shift_image),2)))
    tran_images[j,:,:] = tran_image
    test_movie.set_data(tran_image)
    #test_movie.text("frame: " + str(j))
    return test_movie,

def shift_crop_dir(x,j):
    match x:
        case "R":
            low_lim_shift = low_lim + j
            upp_lim_shift = upp_lim + j
            shifted_image = test_image[low_lim_shift:upp_lim_shift, low_lim:upp_lim]
            return shifted_image
        case "U":
            low_lim_shift = low_lim - j
            upp_lim_shift = upp_lim - j
            shifted_image = test_image[low_lim:upp_lim, low_lim_shift:upp_lim_shift]
            return shifted_image
        case "L":
            low_lim_shift = low_lim - j
            upp_lim_shift = upp_lim - j
            shifted_image = test_image[low_lim_shift:upp_lim_shift, low_lim:upp_lim]
            return shifted_image
        case "D":
            low_lim_shift = low_lim + j
            upp_lim_shift = upp_lim + j
            shifted_image = test_image[low_lim:upp_lim, low_lim_shift:upp_lim_shift]
            return shifted_image

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
crop_image = reff_image[low_lim:upp_lim, low_lim:upp_lim]
ax = plt.axes(xlim=(0, upp_lim-low_lim), ylim=(0,upp_lim-low_lim))
test_movie = ax.imshow(crop_image, cmap='gray')
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                            frames=shifts, interval=20, blit=True) 
anim.save('basic_animation.gif', fps=5)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

plt.show()

plt.plot(L2_norm)
plt.show()