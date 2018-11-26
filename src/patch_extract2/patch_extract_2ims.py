 


import cv2
import h5py
import scipy.io as sio
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report
from sklearn.externals import joblib
import sklearn.preprocessing as pp
from osgeo import gdal
import deb
from sklearn.externals import joblib
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-w', '--window_len', default=32,help="Path to weights file to load source model for training classification/adaptation")
ap.add_argument('-s', '--step', default=32,help="Path to weights file to load source model for training classification/adaptation")
ap.add_argument('-wpx', '--wildfire_min_pixel_percentage', default=-1, \
	help="Extract patches which have the wildfire class on them only. Use 'any' for at least 1px")
ap.add_argument('-at', '--all_train', default=False,help="Modify train/Test mask so that almost everything is used for training")
ap.add_argument('-ds', '--dataset', default="para",help="Modify train/Test mask so that almost everything is used for training")

a = ap.parse_args()

def path_configure(dataset,source='tiff',train_test_mask='TrainTestMask.png'):

	path={}
	if dataset=='para':
		path['data']='../../data/AP1_Para/'
		path['raster']=path['data']+'L8_224-66_ROI_clip.tif'
		path['label']=path['data']+'labels.tif'
		path['bounding_box']=path['data']+'bounding_box_pa_clip.tif'
	elif dataset=='acre':
		path['data']='../../data/AP2_Acre/'
		if source=='tiff':
			path['raster']=path['data']+'L8_002-67_ROI.tif'
		elif source=='matlab':
			path['raster']='/home/lvc/Jorg/igarss/wildfire_fcn/data/AP2_Acre/dataset.h5'
			
			#path['raster']='/home/lvc/Jorg/igarss/wildfire_fcn/data/AP2_Acre/acre_matched.mat'
			#path['raster']=path['data']+'acre_matched.mat'
		path['label']=path['data']+'labels.tif'
		path['bounding_box']=path['data']+'bounding_box_clip.tif'
	path['train_test_mask']=path['data']+train_test_mask
	return path	

def unique_count_print(x):
	deb.prints(np.unique(x,return_counts=True))
def im_load(path,dataset):
	im = gdal.Open(path['raster'])
	im = np.array(im.ReadAsArray())
	im = np.transpose(im, (1, 2, 0))
	
	# debug
	# deb.prints(np.count_nonzero(im[im>30000]))
	# im_=im[:,:,0].copy()
	# deb.prints(np.count_nonzero(im_>30000))
	# deb.prints(im_.shape)
	# im_[im_<=30000]=0
	# im_[im_>30000]=250	
	# im_=im_.astype(np.uint8)

	# cv2.imwrite("toobright.png",im_)
	im = im.astype(np.float32)

	if dataset=='para':
		im = im[0:-1,0:-1,:] # Eliminate non used pixels
	elif dataset=='acre':
		im = im[0:-1,:,:]
	return im

def mask_label_load(path,im,flatten=False,all_train=False):

	deb.prints(path['train_test_mask'])
	mask=cv2.imread(path['train_test_mask'],0).astype(np.uint8)
	unique_count_print(mask)
	label=cv2.imread(path['label'],-1).astype(np.uint8)
	label[label==2]=1 # Only use 2 classes
	label=label+1 # 0 is for background

	bounding_box=cv2.imread(path['bounding_box'],-1).astype(np.uint8)
	channel_n=6
	chans=range(channel_n)
	for chan in chans:
		bounding_box[im[:,:,chan]==32767]=0

	#mask[mask==255]=1
	#mask=mask+1

	if all_train==True:
		mask.fill(1)

	mask[bounding_box==0]=0 # Background. No data
	label[bounding_box==0]=0 # Background. No data
	if flatten==True:
		mask=mask.reshape(-1)
		label=label.reshape(-1)
	# Not quite necessary to do this but more informative

	return mask,label,bounding_box

def view_as_windows_flat(im,window_shape,step=1):
	info={}
	patch=view_as_windows(im,window_shape,step)
	windows_shape=patch.shape
	patch=np.squeeze(patch)
	patch=np.reshape(patch,(patch.shape[0]*patch.shape[1],)+patch.shape[2:])
	deb.prints(patch.shape)
	return patch,windows_shape

one_image=False
# Load first image
dataset=a.dataset

#im_path='../../data/AP2_Acre/L8_002-67_ROI.tif'

path=path_configure(dataset,source='tiff')
im=im_load(path,dataset)
mask,label,bounding_box=mask_label_load(path,im)

deb.prints(im.shape)
deb.prints(mask.shape)
deb.prints(label.shape)
deb.prints(np.unique(bounding_box))
mask=mask.astype(np.uint8)
label=label.astype(np.uint8)
bounding_box=bounding_box.astype(np.uint8)
im=im.astype(np.float32)

channel_n=im.shape[2]
deb.prints(channel_n)
# ======== Normalize ==============


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
def stats_print(x):
    print(np.min(x),np.max(x),np.average(x),x.dtype)
# stats_print(im)
# deb.prints(im.shape)
# h,w,chans=im.shape
# im_flat=np.reshape(im,(h*w,chans))
# bounding_box_flat=np.reshape(bounding_box,-1)
# stats_print(im_flat[bounding_box_flat!=0])
def scaler_from_im_fit(im,mask,debug=0):
	h,w,chans=im.shape
	im=np.reshape(im,(h*w,chans))
	mask=np.reshape(mask,-1)
	deb.prints(im.shape)
	# Pick source pixels
	im_train=im[mask==1]

	# Correct for saturated values
	#avg=np.average(im_train)
	#n=32767
	#im_train[im_train==n]=np.average(im_train[im_train!=n])
	stats_print(im_train)
	stats_print(im_train[:,0])
	stats_print(im_train[:,1])
	stats_print(im_train[:,2])
	stats_print(im_train[:,3])
	stats_print(im_train[:,4])
	stats_print(im_train[:,5])

	if debug>=2:
		
		deb.prints(np.count_nonzero(im_train[im_train>30000]))

		deb.prints(np.count_nonzero(im_train[im_train>32760]))
		deb.prints(np.count_nonzero(im_train[:,0][im_train[:,0]>32760]))
		deb.prints(np.count_nonzero(im_train[:,1][im_train[:,1]>32760]))
		deb.prints(np.count_nonzero(im_train[:,2][im_train[:,2]>32760]))
		deb.prints(np.count_nonzero(im_train[:,3][im_train[:,3]>32760]))
		deb.prints(np.count_nonzero(im_train[:,4][im_train[:,4]>32760]))
	

	deb.prints(im.shape)
	# Fit on train area
	scaler=MinMaxScaler()
	scaler.fit(im_train)

	# Transform on whole image
	im_train=scaler.transform(im_train)
	im=scaler.transform(im)
	im=np.reshape(im,(h,w,chans))
	deb.prints(im.shape)
	# 
	stats_print(im_train)
	return im,scaler

cv2.imwrite("im1.png",im[:,:,0:3])
cv2.imwrite("im2.png",im[:,:,3:6])

#import tifffile as tiff

#imsave("im.tif", im)
if one_image==True:
	im,scaler= scaler_from_im_fit(im,mask)
else:
	im,scaler= scaler_from_im_fit(im,bounding_box)
stats_print(im)


def unnormalize(im,scaler):
    h,w,chans=im.shape
    im=np.reshape(im,(h*w,chans))
    im=scaler.inverse_transform(im)
    #stats_print(im)
    return np.reshape(im,(h,w,chans))

im_rescale=unnormalize(im,scaler)
cv2.imwrite("im_rescale.png",im_rescale[:,:,0:3])
# ========= Mask ===============

def im_apply_mask(im,mask,channel_n):
	im_train=im.copy()
	im_test=im.copy()
	
	for band in range(0,channel_n):
		im_train[:,:,band][mask!=1]=-2
		im_test[:,:,band][mask!=2]=-2
	deb.prints(im_train.shape)
	return im_train,im_test
def label_apply_mask(im,mask): 
	im=im.astype(np.uint8) 
	im_train=im.copy() 
	im_test=im.copy() 
	 
	mask_train=mask.copy() 
	mask_train[mask==2]=0 
	mask_test=mask.copy() 
	mask_test[mask==1]=0 
	mask_test[mask==2]=1 
 
	deb.prints(im.shape) 
	deb.prints(mask_train.shape) 
 
	deb.prints(im.dtype) 
	deb.prints(mask_train.dtype) 
	 
	im_train=cv2.bitwise_and(im,im,mask=mask_train) 
	im_test=cv2.bitwise_and(im,im,mask=mask_test) 
 
 
	#im_train[t_step,:,:,band][mask!=1]=-1 
	#im_test[t_step,:,:,band][mask!=2]=-1 
	deb.prints(im_train.shape) 
	return im_train,im_test 
data={'train':{},'test':{}}
# These images and labels are already isolated between train and test areass
data['train']['im'], data['test']['im'] = im_apply_mask(im,mask,channel_n)
data['train']['label'], data['test']['label'] = label_apply_mask(label,mask)
data['train']['mask'], data['test']['mask'] = label_apply_mask(label,mask)

# ========= Extract patches ==========
def patches_from_domain_gather(patches,mask,domain,axis=(1,2),mask_min_pixel_percentage=0.5):
		if mask_min_pixel_percentage=="any":
			return patches[np.any(mask==domain,axis=axis),::]
		else:
			pixel_limit=int(patches.shape[1]*patches.shape[2]*mask_min_pixel_percentage)
			return patches[np.count_nonzero(mask==domain,axis=axis)>pixel_limit,::]
def patches_from_subset(subset,data,window_shape,patches_step, \
	mask_min_pixel_percentage, wildfire_min_pixel_percentage=-1):
	subset['im'],_=view_as_windows_flat(data['im'],window_shape,step=patches_step)
	subset['mask'],_=view_as_windows_flat(data['mask'],(window_len,window_len),step=patches_step)
	subset['label'],_=view_as_windows_flat(data['label'],(window_len,window_len),step=patches_step)
	
	subset['im']=patches_from_domain_gather(subset['im'],subset['mask'], \
		1,mask_min_pixel_percentage=mask_min_pixel_percentage)
	subset['label']=patches_from_domain_gather(subset['label'],subset['mask'], \
		1,mask_min_pixel_percentage=mask_min_pixel_percentage)
	if wildfire_min_pixel_percentage>0:
		subset['im']=patches_from_domain_gather(subset['im'], \
			subset['mask'],1,mask_min_pixel_percentage=wildfire_min_pixel_percentage)
		subset['label']=patches_from_domain_gather(subset['label'], \
			subset['mask'],1,mask_min_pixel_percentage=wildfire_min_pixel_percentage)
	return subset


from  skimage.util import view_as_windows

window_len=a.window_len
channel_n=6
#patches_step=int(window_len/3)
#patches_step=int(window_len)
patches_step=a.step
deb.prints(patches_step)
window_shape=(window_len,window_len,channel_n)

deb.prints(np.unique(data['train']['label'],return_counts=True))
deb.prints(np.unique(data['test']['label'],return_counts=True))

patches={'train':{},'test':{}}
patches['train']=patches_from_subset(patches['train'],data['train'], \
	window_shape,patches_step,mask_min_pixel_percentage=0.5, \
	wildfire_min_pixel_percentage=a.wildfire_min_pixel_percentage)
patches['test']=patches_from_subset(patches['test'],data['test'], \
	window_shape,patches_step,mask_min_pixel_percentage="any", \
	wildfire_min_pixel_percentage=a.wildfire_min_pixel_percentage)


deb.prints(patches['train']['im'].shape)
deb.prints(patches['test']['im'].shape)

folder="compact/"+dataset+"/"
np.save(folder+"train_im.npy",patches['train']['im'])
np.save(folder+"train_label.npy",patches['train']['label'])
np.save(folder+"test_im.npy",patches['test']['im'])
np.save(folder+"test_label.npy",patches['test']['label'])

joblib.dump(scaler, 'scaler_'+dataset+'.joblib') 
assert 1==2

# ============ END OF SCRIPT ======================= #

def patches_store(patches,path):
		for idx in range(patches.shape[0]):
			np.save(path+"patches"+str(idx)+".npy",patches[idx])

deb.prints(patches['im'].shape)
if one_image==True:
	patches['target']={}
	patches['source']={}


	# I'm taking all sourcxe patches, but masking for normalizing
	patches['target']['im']=patches_from_domain_gather(patches['im'],patches['mask'],2)
	patches['source']['im']=patches_from_domain_gather(patches['im'],patches['mask'],1)

	patches['target']['mask']=patches_from_domain_gather(patches['mask'],patches['mask'],2)
	patches['source']['mask']=patches_from_domain_gather(patches['mask'],patches['mask'],1)

	patches['target']['label']=patches_from_domain_gather(patches['label'],patches['mask'],2)
	patches['source']['label']=patches_from_domain_gather(patches['label'],patches['mask'],1)

	del patches['im']
	del patches['label']
	del patches['mask']


	#patches['target']['mask']=patches_domain_gather(patches['mask'],patches['mask'],2)
	#patches['source']['im']=patches_domain_gather(patches['im'],1)

	#patches['mask'][np.any(patches['mask']==2,axis=(1,2)),::]
	#patches['source']['mask']=patches['mask'][np.any(patches['mask']==1,axis=(1,2)),::]
	#del patches['mask']

	deb.prints(patches['target']['im'].shape)
	deb.prints(patches['source']['im'].shape)


	

	patches_store(patches['source']['im'],"patches/source/im/")
	patches_store(patches['target']['im'],"patches/target/im/")

	patches_store(patches['source']['mask'],"patches/source/mask/")
	patches_store(patches['target']['mask'],"patches/target/mask/")

	patches_store(patches['source']['label'],"patches/source/label/")
	patches_store(patches['target']['label'],"patches/target/label/")
else:
	patches['im']=patches_from_domain_gather(patches['im'],patches['bounding_box'],1)
	patches['mask']=patches_from_domain_gather(patches['mask'],patches['bounding_box'],1)
	patches['label']=patches_from_domain_gather(patches['label'],patches['bounding_box'],1)
	patches_store(patches['im'],"patches/"+dataset+"/im/")

	patches_store(patches['mask'],"patches/"+dataset+"/mask/")

	patches_store(patches['label'],"patches/"+dataset+"/label/")
