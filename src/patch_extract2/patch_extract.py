 


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
	im = im.astype(np.float32)

	if dataset=='para':
		im = im[0:-1,0:-1,:] # Eliminate non used pixels
	elif dataset=='acre':
		im = im[0:-1,:,:]
	return im

def mask_label_load(path,flatten=False):

	deb.prints(path['train_test_mask'])
	mask=cv2.imread(path['train_test_mask'],0).astype(np.uint8)
	unique_count_print(mask)
	label=cv2.imread(path['label'],-1).astype(np.uint8)
	label[label==2]=1 # Only use 2 classes
	label=label+1 # 0 is for background

	bounding_box=cv2.imread(path['bounding_box'],-1).astype(np.uint8)
	#mask[mask==255]=1
	#mask=mask+1
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

dataset='acre'


im_path='../../data/AP2_Acre/L8_002-67_ROI.tif'

path=path_configure(dataset,source='tiff',train_test_mask='train_test_mask_ac_target.png')
mask,label,bounding_box=mask_label_load(path)
im=im_load(path,dataset)
deb.prints(im.shape)
deb.prints(mask.shape)
deb.prints(label.shape)

mask=mask.astype(np.uint8)
label=label.astype(np.uint8)
im=im.astype(np.float32)


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
def scaler_from_im_fit(im,mask):
	h,w,chans=im.shape
	im=np.reshape(im,(h*w,chans))
	mask=np.reshape(mask,-1)
	deb.prints(im.shape)
	# Pick source pixels
	im_train=im[mask==1]
	stats_print(im_train)
	deb.prints(im.shape)
	# Fit on train area
	scaler=StandardScaler()
	scaler.fit(im_train)

	# Transform on whole image
	im=scaler.transform(im)
	im=np.reshape(im,(h,w,chans))
	deb.prints(im.shape)
	# 
	return im,scaler

cv2.imwrite("im1.png",im[:,:,0:3])
cv2.imwrite("im2.png",im[:,:,3:6])

im,scaler= scaler_from_im_fit(im,mask)
stats_print(im)
# ========= Extract patches ==========
from  skimage.util import view_as_windows

window_len=128
channel_n=6
patches_step=int(window_len/3)
deb.prints(patches_step)
window_shape=(window_len,window_len,channel_n)

patches={}
patches['im'],_=view_as_windows_flat(im,window_shape,step=patches_step)
patches['mask'],_=view_as_windows_flat(mask,(window_len,window_len),step=patches_step)
patches['label'],_=view_as_windows_flat(label,(window_len,window_len),step=patches_step)

deb.prints(patches['im'].shape)
patches['target']={}
patches['source']={}

def patches_from_domain_gather(patches,mask,domain,axis=(1,2)):
	return patches[np.all(mask==domain,axis=axis),::]

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


def patches_store(patches,path):
	for idx in range(patches.shape[0]):
		np.save(path+"patches"+str(idx)+".npy",patches[idx])

patches_store(patches['source']['im'],"patches/source/im/")
patches_store(patches['target']['im'],"patches/target/im/")

patches_store(patches['source']['mask'],"patches/source/mask/")
patches_store(patches['target']['mask'],"patches/target/mask/")

patches_store(patches['source']['label'],"patches/source/label/")
patches_store(patches['target']['label'],"patches/target/label/")

joblib.dump(scaler, 'scaler.joblib') 