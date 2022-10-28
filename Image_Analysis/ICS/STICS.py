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
import numpy as np                  # This library is essential for manipulating data.
import matplotlib.pyplot as plt     # This library is essential for plotting data.
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import numpy as np

# CUSTOM FUNCTIONS ==================================================
#====================================================================
"""  """
def create_video(images):

    fig = plt.figure()
    im = plt.imshow(images[0,:,:], cmap = plt.cm.gray)

    def animate(t):
        image = images[t,:,:]
        im.set_array(image)
        return im, 

    anim = FuncAnimation(
        fig,
        animate,
        frames = 100,
        interval = 1000/30,
        blit = True
    )

    plt.show()

    return anim

def ftrans_video(images):
    image_dims = np.shape(images)
    FFT_images = np.zeros(image_dims,dtype=np.complex_)
    for i in range(image_dims[0]):
        Ixy = images[i,:,:]
        FFT_Ixy = np.fft.fftshift(np.fft.fft2(Ixy))
        FFT_images[i, :, :] = FFT_Ixy

    return FFT_images

def show_kspace(images):
    image_dims = np.shape(images)
    FFT_images = np.zeros([image_dims[0], image_dims[1], image_dims[2]])
    for i in range(image_dims[0]):
        test_images = images[i,:,:]
        dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(test_images))
        FFT_images[i, :, :] = np.log(abs(dark_image_grey_fourier))
    
    return FFT_images

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

def magnify_images(images, M):
    
    # Define magnifcation of images to simulate by cropping: 
    image_dim = np.size(images[0,:])
    sim_MAG = M # simulating extra 60x magnification.
    beta = image_dim/2
    alph = (1/np.sqrt(sim_MAG))*beta
    low_lim = int(np.floor(beta - alph))
    upp_lim = int(np.ceil(beta + alph))
    image_dim = upp_lim - low_lim

    # magnify images:
    magnified_images = images[:, low_lim:upp_lim, low_lim:upp_lim]
    return magnified_images

def kspace_correlation(FFT_images):
    # From paper this is:
    # r(k;s,c) = 1/(P-s) sum from c=0 to P-s-1 <i(k;c)i*(k;c+s)>
    # r = kspace autocorrelation
    # k = kspace vector
    # s = delay coordinate  (tau)
    # c = time coordinate   (t)
    # P = total number of time points (images)
    # i = fourier transform of image series
    # i* = complex conjugate of i

    image_dims = np.shape(FFT_images)
    FFT_conjs = np.conj(FFT_images)
    r = np.zeros(image_dims, dtype=np.complex_)
    P = image_dims[0]

    for s in range(P):
        sum_Is = 0
        for c in range(P-s-1):
            sum_Is = sum_Is + FFT_images[c,:,:]*FFT_conjs[c+s,:,:]
        r[s,:,:] = (1/(P-s))*sum_Is

    kcorr_images = r
    return kcorr_images
    
def gspace_correlation(images):

    Ixyt = images
    FFT_Ixyt = np.fft.fftn(Ixyt) # Fourier ransform the image
    PSD_Ixyt = np.fft.ifftn(FFT_Ixyt*np.conj(FFT_Ixyt)) # Power spectral density
    G_Ixyt = np.fft.fftshift(PSD_Ixyt) # Autocorrelaton (G_IXY) 
    return G_Ixyt
    
"""     image_dims = np.shape(images)
    G_Ixyt = np.zeros(image_dims,dtype=np.complex_)
    im_conjs = np.conj(images)
    P = image_dims[0]
    for s in range(P):
        sum_Is = 0
        for c in range(P-s-1):
            sum_Is = sum_Is + images[c,:,:]*im_conjs[c+s,:,:]
        G_Ixyt[s,:,:] = (1/(P-s))*sum_Is

    return G_Ixyt """
        
"""     for i in range(image_dims[0]):
        Ixy = 0
        FFT_Ixy = np.fft.fft2(Ixy) # Fourier ransform the image
        PSD_Ixy = np.fft.ifft2(FFT_Ixy*np.conj(FFT_Ixy)) # Power spectral density
        G_Ixy = np.fft.fftshift(PSD_Ixy) # Autocorrelaton (G_IXY) 
    return G_Ixy """

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
    image_dimx = np.size(image[0,:])
    image_dimy = np.size(image[:,0])
    
    # Make data.
    X = np.arange(-1*image_dimx/2, image_dimx/2, 1)
    Y = np.arange(-1*image_dimy/2, image_dimy/2, 1)
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

showplots = True

data_path =  "C:/Users/Cameron/Documents/GitHub/Tutorial/data/"
data_fold = "2022-07-PDMS+TIRF/"
#data_fold = "2022-09-Heart_Cells/"
#data_fold = "2022-07-Patterns/"

test_name = "iCM-point4-beads-31fps_cropped.tif"
#test_name = "TIRF.tif"

test_file = data_path + data_fold + test_name

# Grab the test and reference images
global test_images
test_images = io.imread(test_file)
if showplots:
    anim = create_video(test_images)

# NORMALIZE IMAGES ==================================================
#====================================================================

# Normalize (make everything go from 0-1)
test_images = normalize_images(test_images)

# CROP IMAGES =======================================================
#====================================================================

# Crop images to the smallest possible dimension to make a sqaure
test_images = crop_images(test_images)
gcorr_images = gspace_correlation(test_images)
gcorr_anim = create_video(np.log(np.abs(gcorr_images)))
# COMPUTE FFT =======================================================
#====================================================================

FFT_images = ftrans_video(test_images)
FFT_vid = show_kspace(test_images)
FFT_anim = create_video(FFT_vid)

# COMPUTE KSPACE AUTOCORRELATION ====================================
#====================================================================

kcorr_images = kspace_correlation(FFT_images)
kcorr_anim = create_video(np.log(np.abs(kcorr_images)))
real_anim = create_video(np.log(np.power(np.real(kcorr_images),2)))
imag_anim = create_video(np.log(np.power(np.imag(kcorr_images),2)))

#tif.imsave('a.tif', kcorr_images, bigtiff=True)

print(kcorr_anim)
# MAGNIFY IMAGES ====================================================
#====================================================================

""" magnification = 60 # Nx magnification
test_images = magnify_images(test_images, magnification)
anim = create_video(test_images) """

# WAVELET  TRANSFORMATION ===========================================
#====================================================================

""" image_dim = np.size(test_images,1)
wave_mats = np.zeros([int(np.sqrt(image_dim)), image_dim, image_dim])
wave_mat = compute_wkernel(test_images, 20)
wave_mat = normalize_intensities(wave_mat)
Tab = test_images*wave_mat
plt.imshow(Tab)
plt.show() """

# COMPUTE AUTOCORRELATION ===========================================
#====================================================================

#FFT_Ixy_test, PSD_Ixy_test, G_Ixy_test = compute_G(test_images)

# SHOW AUTOCORRELATIONS =============================================
#====================================================================

#G_heatmaps(test_images, G_Ixy_test)

# SHOW AUTOCORRELATION SURFACES + CONTOURS ==========================
#====================================================================

#G_surfaces(test_images, G_Ixy_test)

""" pixel2difflim = 2
lam = 488e-9
NA = 1.49
G_profiles(test_images, G_Ixy_test, lam, NA, pixel2difflim) """

# COMPUTE NUMBER OF PARTICLES IN IMAGE ==============================
#====================================================================

#bead_num_TIRF   = str(compute_N(G_Ixy_test, lam, NA, pixel2difflim))

# Format a print statement
#print("TIRF     N = " + bead_num_TIRF)
