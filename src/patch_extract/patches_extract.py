 
from __future__ import division
import os
import math
#import json
import random
#import pprint
#import scipy.misc
import numpy as np
from time import gmtime, strftime
#from osgeo import gdal
import glob
#from skimage.transform import resize
#from sklearn import preprocessing as pre
#import matplotlib.pyplot as plt
import cv2
import pathlib
#from sklearn.feature_extraction.image import extract_patches_2d
#from skimage.util import view_as_windows
import sys
import pickle
# Local
import deb
import argparse
from sklearn.preprocessing import StandardScaler
from skimage.util import view_as_windows

from PIL import Image
from osgeo import gdal

#Input configuration
parser = argparse.ArgumentParser(description='')
parser.add_argument('-pl','--patch_len', dest='patch_len', type=int, default=5, help='# timesteps used to train')
parser.add_argument('-ps','--patch_step', dest='patch_step', type=int, default=0, help='Debug')
parser.add_argument('--band_n', dest='band_n', type=int, default=6, help='Debug')
parser.add_argument('--path', dest='path', default="../../data/AP1_Para/", help='Data path')
parser.add_argument('--class_n', dest='class_n', type=int, default=2, help='Class number')
parser.add_argument('-bs','--balance_samples_per_class', dest='balance_samples_per_class',type=int,default=None, help="Class number. 'local' or 'remote'")
parser.add_argument('-ttmn','--train_test_mask_name', dest='train_test_mask_name',default="TrainTestMask.tif", help="Class number. 'local' or 'remote'")
parser.add_argument('-psv','--patches_save', dest='patches_save',default=True, help="Patches npy store")

a = parser.parse_args()
np.set_printoptions(suppress=True)

#class Patches(object):
#	def __init__(self,a):
#		self.a=a
#		deb.prints(self.a)
#	def load_image():
#	def train_test_split():
#	def normalize():


#def load_image():
def print_min_max_avg(in_):
	print(np.min(in_),np.max(in_),np.average(in_))
	 
def normalize(im, mask):
	h,w,channels=im.shape
	im_flat=np.reshape(im,(h*w,channels))
	mask_flat=np.reshape(mask,-1)

	deb.prints(im_flat.shape)
	deb.prints(mask_flat.shape)
	train_flat=im_flat[mask_flat==255,:]

	deb.prints(train_flat.shape)

	print_min_max_avg(train_flat)

	scaler=StandardScaler()
	scaler.fit(train_flat)
	train_norm_flat=scaler.transform(train_flat)

	print_min_max_avg(train_norm_flat)

	im_norm_flat=scaler.transform(im_flat)
	im_norm=np.reshape(im_norm_flat,(h,w,channels))
	deb.prints(im_norm.shape)
	

	print("FINISHED NORMALIZING, RESULT:")
	print_min_max_avg(im_norm)
	return im_norm

def patches_extract(im,patch_len,band_n,step,debug=1):
	deb.prints(band_n)
	if band_n != -1:
		window_shape=(patch_len,patch_len,band_n)
	else:
		window_shape=(patch_len,patch_len)
	if debug:
		deb.prints(window_shape)
		deb.prints(im.shape)
	return np.squeeze(view_as_windows(im, window_shape, step=step))

def im_reconstruct_from_patches(patches,a):
	patches=patches.reshape((a.patches_info['rows'], \
		a.patches_info['cols']) + patches.shape[1:])
	out=np.zeros(a.im_shape[:-1])
	for row in range(a.patches_info['rows']):
		for col in range(a.patches_info['cols']):
			out[row*a.patch_step:(row+1)*a.patch_step, \
				col*a.patch_step:(col+1)*a.patch_step]= \
				patches[row,col]
	return out

if __name__ == '__main__':
	path={}
	path['raster']=a.path+'L8_224-66_ROI_clip.tif'
	path['label']=a.path+'labels.tif'
	path['train_test_mask']=a.path+'TrainTestMask.png'
	# Read image
	im = gdal.Open(path['raster'])
	im = np.array(im.ReadAsArray())
	im = np.transpose(im, (1, 2, 0))
	im = im[0:-1,0:-1,:] # Eliminate non used pixels
	im = im.astype(np.float32)
	deb.prints(im.shape)
	deb.prints(im.dtype)

	a.band_n=im.shape[2]
	deb.prints(a.band_n)
	masks={}
	masks['train_test']=cv2.imread(path['train_test_mask'],0).astype(np.uint8)
	masks['label']=cv2.imread(path['label'],-1).astype(np.uint8)
	masks['label'][masks['label']==2]=1 # Only use 2 classes
	cv2.imwrite("label_original.png",masks['label']*255)
	deb.prints(masks['train_test'].shape)
	deb.prints(masks['label'].shape)
	
	# ============ Normalize ========================
	im = normalize(im, masks['train_test'])
	deb.prints(im.shape)

	# =========== Extract patches  ===================
	patches={}
	patches['im'] = patches_extract(im,a.patch_len,a.band_n,a.patch_step)
	patches['label'] = patches_extract(masks['label'],a.patch_len,-1,a.patch_step)
	patches['train_test'] = patches_extract(masks['train_test'],a.patch_len,-1,a.patch_step)
	
	a.im_shape=im.shape
	a.patches_info={}
	a.patches_info['rows']=patches['label'].shape[0]
	a.patches_info['cols']=patches['label'].shape[1]

	deb.prints(patches['label'].shape)
	deb.prints(patches['im'].shape)
	
	#print()
	patches['im']=patches['im'].reshape((a.patches_info['rows']*a.patches_info['cols'],)+patches['im'].shape[2:])
	patches['label']=patches['label'].reshape((a.patches_info['rows']*a.patches_info['cols'],)+patches['label'].shape[2:])
	patches['train_test']=patches['train_test'].reshape((a.patches_info['rows']*a.patches_info['cols'],)+patches['train_test'].shape[2:])
	
	# ====== Reconstruct label. Just for assertion here. =====
	label_reconstruct = im_reconstruct_from_patches(patches['label'],a)
	deb.prints(patches['im'].shape)
	deb.prints(label_reconstruct.shape)
	deb.prints(np.unique(label_reconstruct))
	cv2.imwrite("label_reconstruct.png",label_reconstruct*255)

	id_train_test=np.zeros(patches['im'].shape[0])
	# ====== Train test split
	for idx in range(patches['im'].shape[0]):
		if np.all(patches['train_test'][idx]>=1):
			id_train_test[idx]=True
		else:
			id_train_test[idx]=False

	names={"train":{},"test":{}}
	names['train']['im']='train_ims.npy'


	patches_im_name="patches_im.npy"
	patches_label_name="patches_label.npy"
	id_train_test_name="id_train_test.npy"
	
	np.save(patches_im_name,patches['im'])
	np.save(patches_label_name,patches['label'])	
	np.save(id_train_test_name,id_train_test)

	deb.prints(id_train_test.shape)
	deb.prints(patches['im'][id_train_test==True].shape)
	deb.prints(patches['im'][id_train_test==False].shape)


	# ============== Store patches into npy


	# Extract
	# Train test split
	#patches=Patches(a)
	#patches.load_image()
	#patches.train_test_split()
	#patches.normalize()
	#patches.extract()


