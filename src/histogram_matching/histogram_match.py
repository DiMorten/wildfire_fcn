 
#from skimage.exposure import cumulative_distribution
#import matplotlib.pylab as plt
#import numpy as np
import sklearn
from matplotlib import pyplot as plt

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


acre_path="../../data/AP2_Acre/L8_224-66_ROI_clip.tif"
acre_im=im_load(acre_path.ravel())
para_path="../../data/AP2_Acre/L8_002-67_ROI.tif"
para_im=im_load(para_path.ravel())

vals,ecdf=ecdf(acre_im)
vals,ecdf=ecdf(para_im)
