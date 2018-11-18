 
#from skimage.exposure import cumulative_distribution
#import matplotlib.pylab as plt
import numpy as np
import sklearn
from osgeo import gdal
import deb
from PIL import Image
#from matplotlib import pyplot as plt

def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf


def hist_matching(c, c_t, im):
	'''
	c: CDF of input image computed with the function cdf()
	c_t: CDF of template image computed with the function cdf()
	im: input image as 2D numpy ndarray
	returns the modified pixel values
	''' 
	pixels = np.arange(256)
	# find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of   
	# the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
	new_pixels = np.interp(c, c_t, pixels) 
	im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype(np.uint8)
	return im

def im_load(path):

	# Read image
	im = gdal.Open(path)
	im = np.array(im.ReadAsArray())
	im = np.transpose(im, (1, 2, 0))

	im = im.astype(np.float32)
	deb.prints(im.shape)
	deb.prints(im.dtype)

	return im

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)
acre_path="../../data/AP2_Acre/L8_002-67_ROI.tif"
acre_im=im_load(acre_path)
#acre_avg=acre_im[:][acre_im[:]!=32767]
deb.prints(np.average(acre_im))
print(acre_im.shape)
#acre_shape=acre_im.shape
#acre_im=np.reshape(acre_im,-1)
para_path="../../data/AP1_Para/L8_224-66_ROI_clip.tif"
para_im=im_load(para_path)
#para_avg=para_im[:][para_im[:]!=32767]

deb.prints(np.average(para_im))
print(para_im.shape)
#para_im=para_im.shape
#para_im=np.reshape(para_im,-1)
#vals,ecdf=ecdf(acre_im.ravel())
#vals,ecdf=ecdf(para_im.ravel())
matched=hist_match(para_im,acre_im)
print(matched.shape)
deb.prints(np.average(matched))

