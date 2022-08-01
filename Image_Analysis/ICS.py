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
from matplotlib import animation

# CUSTOM FUNCTIONS ==================================================
#====================================================================

import math
import cv2
import numpy as np

# Following three images are from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

# TODO: How necessary is this part? Maybe only useful if the image is not a square.
def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

# Crop an image about its centre point
def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

# Initialization function: plot the background of each frame
def init():
    test_movie.set_data(crop_image - crop_image)
    return test_movie,

# Animation function.  This is called sequentially
def animate_shifts(j):
    # directions = [R,U,L,D]    |    direction_repeats = [1,1,3,3,4,4,5,5,...,Nshift,Nshift]
    direct = directions[np.mod(j,len(directions)-1)]
    shift_image, low_lim_x, upp_lim_x, low_lim_y, upp_lim_y = shift_crop_dir(direct,j)
    tran_image = crop_image*shift_image
    L2_norm[j,0] = np.sum(crop_image*shift_image)
    R_vals[j] = np.sqrt(np.power(low_lim_x-low_lim, 2) + np.power(low_lim_y-low_lim, 2))
    #tran_image = np.power((crop_image-shift_image),2)
    #L2_norm[j] = np.sqrt(np.sum(np.power((crop_image-shift_image),2)))
    tran_images[j,:,:] = tran_image
    test_movie.set_data(tran_image)
    return test_movie,

# Helper function for deciding how to shift the matrix
def shift_crop_dir(x, j):
    match x:
        case "R":
            new_low_lim_x = low_lim_x + j
            new_upp_lim_x = upp_lim_x + j
            new_low_lim_y = low_lim_y
            new_upp_lim_y = upp_lim_y
            shifted_image = test_image[new_low_lim_x:new_upp_lim_x, new_low_lim_y:new_upp_lim_y]
            return shifted_image, new_low_lim_x, new_upp_lim_x, new_low_lim_y, new_upp_lim_y
        case "U":
            new_low_lim_x = low_lim_x 
            new_upp_lim_x = upp_lim_x 
            new_low_lim_y = low_lim_y - j
            new_upp_lim_y = upp_lim_y - j
            shifted_image = test_image[new_low_lim_x:new_upp_lim_x, new_low_lim_y:new_upp_lim_y]
            return shifted_image, new_low_lim_x, new_upp_lim_x, new_low_lim_y, new_upp_lim_y
        case "L":
            new_low_lim_x = low_lim_x - j
            new_upp_lim_x = upp_lim_x - j
            new_low_lim_y = low_lim_y
            new_upp_lim_y = upp_lim_y
            shifted_image = test_image[new_low_lim_x:new_upp_lim_x, new_low_lim_y:new_upp_lim_y]
            return shifted_image, new_low_lim_x, new_upp_lim_x, new_low_lim_y, new_upp_lim_y
        case "D":
            new_low_lim_x = low_lim_x 
            new_upp_lim_x = upp_lim_x 
            new_low_lim_y = low_lim_y + j
            new_upp_lim_y = upp_lim_y + j 
            shifted_image = test_image[new_low_lim_x:new_upp_lim_x, new_low_lim_y:new_upp_lim_y]
            return shifted_image, new_low_lim_x, new_upp_lim_x, new_low_lim_y, new_upp_lim_y

# INIT IMAGES =======================================================
#====================================================================

#test_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/UnTIRF+BKGD.tif"
#reff_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/TIRF+BKGD.tif"
test_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/TIRF.tif"
reff_file = "C:/Users/Cameron/Documents/GitHub/Tutorial/Image_Analysis/data/2022-07-PDMS+TIRF/unTIRF.tif"

# Grab the test and reference images
test_image = plt.imread(test_file)
reff_image = plt.imread(reff_file)
# Normalize 
test_image = (test_image - np.min(test_image)) / (np.max(test_image) - np.min(test_image))
reff_image = (reff_image - np.min(reff_image)) / (np.max(reff_image) - np.min(reff_image))
# Image dimension, assumes similar sizes
image_dim = np.size(test_image[0,:])

# INIT SHIFTS =======================================================
#====================================================================

# Define fraction of image boarder to be cropped: A*
crop_frac = 1/2.1
low_lim = int(np.floor(image_dim*crop_frac))
upp_lim = int(np.floor(image_dim*(1-crop_frac)))
low_lim_x = low_lim
upp_lim_x = upp_lim
low_lim_y = low_lim
upp_lim_y = upp_lim
# Define the order of shift spiral
directions = ['R', 'U', 'L', 'D']
# Max number of shifts
num_shifts = int(low_lim/10)
num_angles = num_shifts
# Init transformation matrix ~ movie
tran_images = np.zeros([num_shifts, upp_lim-low_lim, upp_lim-low_lim])
direction_reps = np.zeros([2*num_shifts,1])
# direction_repeats = [1,1,3,3,4,4,5,5,...,Nshift,Nshift]
for i in range(num_shifts):
    loop_shifted_ind = 2*i
    direction_reps[loop_shifted_ind] = i+1
    direction_reps[loop_shifted_ind+1] = i+1
# Init L2 norm matrix
L2_norm = np.zeros([num_shifts, num_angles])
# Init R vector 
R_vals = np.zeros([num_shifts, 1])

dir_counter = 0
step_counter = 0

# MAIN CODE STARTS HERE =============================================
#====================================================================
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
crop_image = reff_image[low_lim:upp_lim, low_lim:upp_lim]
ax = plt.axes(xlim=(0, upp_lim-low_lim), ylim=(0,upp_lim-low_lim))
test_movie = ax.imshow(crop_image)
# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate_shifts, init_func=init,
                            frames=num_shifts, interval=20, blit=True) 

for j in range(num_shifts):
    # directions = [R,U,L,D]    |    direction_repeats = [1,1,3,3,4,4,5,5,...,Nshift,Nshift]
    direct = directions[np.mod(j,len(directions)-1)]
    shift_image, low_lim_x, upp_lim_x, low_lim_y, upp_lim_y = shift_crop_dir(direct,j)
    tran_image = crop_image*shift_image
    L2_norm[j,0] = np.sum(crop_image*shift_image)
    for i in range(num_angles-1):
        rot_image = rotate_image(test_image,360*(i+1)/num_angles)
        rot_image = rot_image[low_lim_x:upp_lim_x, low_lim_y:upp_lim_y]
        L2_norm[j,i] = np.sum(crop_image*rot_image)
    R_vals[j] = np.sqrt(np.power(low_lim_x-low_lim, 2) + np.power(low_lim_y-low_lim, 2))
    #L2_norm[j] = np.sum(crop_image*shift_image)/np.sum(crop_image*crop_image)
    #tran_image = np.power((crop_image-shift_image),2)
    #L2_norm[j] = np.sqrt(np.sum(np.power((crop_image-shift_image),2)))
    tran_images[j,:,:] = tran_image
#anim.save('basic_animation.gif', fps=5)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

plt.show()

plt.imshow(L2_norm)
plt.show()

plt.plot(R_vals)
plt.show()

input()