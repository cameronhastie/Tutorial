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
from skimage import io
import napari
import numpy as np                  # This library is essential for manipulating data.
import matplotlib.pyplot as plt     # This library is essential for plotting data.
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import numpy as np

def normalize_images(images):
    normalized_images = (images - np.min(images)) / (np.max(images) - np.min(images))
    #test_images = (test_images - np.average(test_images)) / np.average(test_images)
    #reff_image = (reff_image - np.average(reff_image)) / np.average(reff_image)
    return normalized_images

def crop_images(images):
    image_dims = np.shape(images)
    min_dim = np.min([image_dims[1], image_dims[2]])
    cropped_images = images[:, 0:min_dim, 0:min_dim]
    return cropped_images

def gspace_correlation(images):

    Itxy = images
    G_Itxy = np.zeros(np.shape(Itxy))
    FFT_t = np.fft.fftn(Itxy[0,:,:]) # Fourier ransform the image at t
    time_points = np.size(images,0)
    for tau in range(time_points):
        FFT_tau = np.fft.fftn(Itxy[tau,:,:]) # Fourier ransform the image at tau
        PSD_tau = np.fft.ifftn(FFT_t*np.conj(FFT_tau)) # Power spectral density
        G_Itxy[tau,:,:] = np.fft.fftshift(PSD_tau) # Autocorrelaton (G_IXY) 

    return G_Itxy

showplots = True

data_path =  "C:/Users/Cameron/Documents/GitHub/Tutorial/data/"
data_fold = "2022-07-PDMS+TIRF/"
#data_fold = "2022-09-Heart_Cells/"
#data_fold = "2022-07-Patterns/"

test_name = "iCM-point4-beads-31fps_sss.tif"
#test_name = "iCM-point4-beads-31fps_cropped.tif"
#test_name = "iCM-point4-beads-31fps_CS.tif"
#test_name = 
#test_name = "TIRF.tif"

test_file = data_path + data_fold + test_name

# Grab the test and reference images
test_images = io.imread(test_file)

# Normalize (make everything go from 0-1)
test_images = normalize_images(test_images)
test_images = crop_images(test_images)

viewer = napari.view_image(test_images)
gcorr_images = np.log(np.abs(gspace_correlation(test_images)))



viewer.add_image(gcorr_images)
napari.run()