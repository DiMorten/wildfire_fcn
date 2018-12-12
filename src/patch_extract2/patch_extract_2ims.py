 


import cv2
import h5py
import scipy.io as sio
import numpy as np
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score,classification_report

import sklearn.preprocessing as pp
from osgeo import gdal
import deb
from sklearn.externals import joblib
import argparse
#from joblib import dump, load
import pathlib
ap = argparse.ArgumentParser()
ap.add_argument('-w', '--window_len', default=32,help="Path to weights file to load source model for training classification/adaptation")
ap.add_argument('-tras', '--train_step', default=32,help="Path to weights file to load source model for training classification/adaptation")
ap.add_argument('-tess', '--test_step', default=32,help="Path to weights file to load source model for training classification/adaptation")
#ap.add_argument('-vals', '--validation_step', default=32,help="Path to weights file to load source model for training classification/adaptation")

ap.add_argument('-wpx', '--wildfire_min_pixel_percentage', default=-1, \
	help="Extract patches which have the wildfire class on them only. Use 'any' for at least 1px")
ap.add_argument('-at', '--all_train', type=bool, default=False,help="Modify train/Test mask so that almost everything is used for training")
ap.add_argument('-ds', '--dataset', default="para",help="Modify train/Test mask so that almost everything is used for training")
ap.add_argument('-of', '--output_folder', default="patches/",help="Modify train/Test mask so that almost everything is used for training")
ap.add_argument('-sp', '--scaler_path', default=None,help="If normalization is to be applied with pre-trained scaler")
ap.add_argument('-val', '--validating', type=bool,default=None,help="If normalization is to be applied with pre-trained scaler")
ap.add_argument('-atst', '--all_test', type=bool, default=False,help="Modify train/Test mask so that almost everything is used for training")
ap.add_argument('-c', '--channel_n', type=int, default=6,help="Modify train/Test mask so that almost everything is used for training")

a = ap.parse_args()


if a.all_train=="True":
	a.all_train=True
elif a.all_train=="False":
	a.all_train=False
if a.validating=="True":
	a.validating=True
if a.all_test=="True":
	a.all_test=True
elif a.all_test=="False":
	a.all_test=False

deb.prints(a.all_train)
deb.prints(a.all_test)
pathlib.Path(a.output_folder).mkdir(parents=True, exist_ok=True)

def stats_print(x):
    print(np.min(x),np.max(x),np.average(x),x.dtype)
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
	elif dataset=='area3':
		path['data']='../../data/vaihinghen/area3/'
		path['raster']=path['data']+'im.tif'
		path['label']=path['data']+'labels.tif'
		path['bounding_box']=path['data']+'bounding_box.tif'
	elif dataset=='area23':
		path['data']='../../data/vaihinghen/area23/'
		path['raster']=path['data']+'im.tif'
		path['label']=path['data']+'labels.tif'
		path['bounding_box']=path['data']+'bounding_box.tif'

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

def mask_label_load(path,im,channel_n,dataset,flatten=False,
	all_train=False,validating=False,all_test=False):

	deb.prints(path['train_test_mask'])
	mask=cv2.imread(path['train_test_mask'],0).astype(np.uint8)
	unique_count_print(mask)
	label=cv2.imread(path['label'],-1).astype(np.uint8)
	if dataset=='acre' or dataset=='para':
		label[label==2]=1 # Only use 2 classes
		label=label+1 # 0 is for background

	bounding_box=cv2.imread(path['bounding_box'],-1).astype(np.uint8)
	
	chans=range(channel_n)
	for chan in chans:
		bounding_box[im[:,:,chan]==32767]=0

	#mask[mask==255]=1
	#mask=mask+1
	print("Mask")
	stats_print(mask)	
	if all_train==True:
		if validating==True:
			mask[mask!=3]=1
		else:
			mask.fill(1)
	if all_test==True:
		if validating==True:
			mask[mask!=3]=2
		else:
			mask.fill(2)
	stats_print(mask)
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
	deb.prints(im.shape)
	deb.prints(windows_shape)
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
mask,label,bounding_box=mask_label_load(path,im,
	a.channel_n,a.dataset,all_train=a.all_train,
	validating=a.validating,all_test=a.all_test)

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

# stats_print(im)
# deb.prints(im.shape)
# h,w,chans=im.shape
# im_flat=np.reshape(im,(h*w,chans))
# bounding_box_flat=np.reshape(bounding_box,-1)
# stats_print(im_flat[bounding_box_flat!=0])
def scaler_from_im_fit(im,mask,dump_name="default",
	scaler_path=None,all_test=False,debug=0):
	h,w,chans=im.shape
	im=np.reshape(im,(h*w,chans))
	mask=np.reshape(mask,-1)
	deb.prints(im.shape)
	deb.prints(np.unique(mask,return_counts=True))
	
	# Pick source pixels
	
	im_train=im[mask==1]

	# Correct for saturated values
	#avg=np.average(im_train)
	#n=32767
	#im_train[im_train==n]=np.average(im_train[im_train!=n])
	stats_print(im_train)

	for chan in range(chans):
		stats_print(im_train[:,chan])	
	
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
	deb.prints(scaler_path)
	if scaler_path==None:
		print("Fitting scaler...")
		scaler=MinMaxScaler()
		scaler.fit(im_train)

		joblib.dump(scaler, dump_name+'.joblib')
	else:
		print("Loading scaler. Scaler path:",scaler_path)
		scaler=joblib.load(scaler_path+'.joblib')
	# Transform on whole image
	im_train=scaler.transform(im_train)
	im=scaler.transform(im)
	im=np.reshape(im,(h,w,chans))
	deb.prints(im.shape)
	# 
	stats_print(im_train)
	return im,scaler

cv2.imwrite("im1.png",im[:,:,0:3])
if a.dataset=='para' or a.dataset=='acre':
	cv2.imwrite("im2.png",im[:,:,3:6])

#import tifffile as tiff

#imsave("im.tif", im)

if one_image==True:
	fit_mask=mask.copy()
else:
	fit_mask=bounding_box.copy()

im,scaler= scaler_from_im_fit(im,fit_mask,
	dump_name=a.output_folder+a.dataset,
	scaler_path=a.scaler_path, all_test=a.all_test)

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

def im_apply_mask(im,mask,channel_n,validating=None):
	im_train=im.copy()
	im_test=im.copy()
	if validating:
		im_val=im.copy()
	for band in range(0,channel_n):
		im_train[:,:,band][mask!=1]=-2
		im_test[:,:,band][mask!=2]=-2
		if validating:
			im_val[:,:,band][mask!=3]=-2
		
	deb.prints(im_train.shape)
	if validating:
		return im_train,im_test,im_val
	else:
		return im_train,im_test,None
def label_apply_mask(im,mask,validating=None): 
	im=im.astype(np.uint8) 
	im_train=im.copy() 
	im_test=im.copy() 
	 
	mask_train=mask.copy() 
	mask_train[mask!=1]=0 
	mask_test=mask.copy() 
	mask_test[mask!=2]=0 
	mask_test[mask==2]=1 
 
	deb.prints(im.shape) 
	deb.prints(mask_train.shape) 
 
	deb.prints(im.dtype) 
	deb.prints(mask_train.dtype) 
	 
	im_train=cv2.bitwise_and(im,im,mask=mask_train) 
	im_test=cv2.bitwise_and(im,im,mask=mask_test) 
 
	if validating:
		mask_val=mask.copy()
		mask_val[mask!=3]=0
		mask_val[mask==3]=1
		im_val=im.copy() 
		im_val=cv2.bitwise_and(im,im,mask=mask_val) 
 
	#im_train[t_step,:,:,band][mask!=1]=-1 
	#im_test[t_step,:,:,band][mask!=2]=-1 
	deb.prints(im_train.shape) 
	if validating:
		return im_train,im_test,im_val 
	else:
		return im_train,im_test,None
def padding_apply(data,window_len,step):
	# Pad for extracting all test patches;
	# Take padded test patches (bounding box=1)
	# but mask the missing values with 0 so that 
	# they are not taken into account. 
	data['mask']=np.pad(data['mask'],(0,window_len),
		'constant',constant_values=2) # So that it thinks is for training
	data['label']=np.pad(data['label'],(0,window_len),
		'constant',constant_values=0) # So that it thinks is for training
	#data['bounding_box']=np.pad(data['bounding_box'],
	#	window_len,'constant',constant_values=1)
	deb.prints(data['im'].shape)
	data['im']=np.pad(data['im'],
		((0,window_len),(0,window_len),(0,0)),
		'constant',constant_values=-2)
	deb.prints(data['im'].shape)
	return data

#def mask_apply_mask(mask):

data={'train':{},'test':{},'val':{}}
# These images and labels are already isolated between train and test areass

if not a.validating:
	mask[mask==3]=1

data['train']['im'], data['test']['im'], data['val']['im'] = im_apply_mask(
	im,mask,channel_n,validating=a.validating)
data['train']['label'], data['test']['label'], data['val']['label'] = label_apply_mask(
	label,mask,validating=a.validating)

data['train']['mask']=mask.copy()
data['test']['mask']=mask.copy()
if a.validating==True:
	data['val']['mask']=mask.copy()

#data['train']=padding_apply(data['train'],a.window_len,
#	a.train_step)
data['test']=padding_apply(data['test'],a.window_len,
	a.test_step)

deb.prints(data['test']['im'].shape)
deb.prints(data['test']['label'].shape)
deb.prints(data['test']['mask'].shape)

'''
def mask_from_subset(mask,subset_id):
	out=mask.copy()
	out[out!=subset_id]=0
	out[out==subset_id]=1
	return out

data['train']['mask']=mask_from_subset(mask,1)
data['test']['mask']=mask_from_subset(mask,2)
data['val']['mask']=mask_from_subset(mask,3)

'''

#data['train']['mask'], data['test']['mask'] = label_apply_mask(label,mask)
deb.prints(np.unique(mask,return_counts=True))
#deb.prints(np.all(data['train']['label']==data['train']['mask']))
#deb.prints(np.all(data['test']['label']==data['test']['mask']))
# ========= Extract patches ==========
def patches_from_domain_gather(patches,mask,domain,axis=(1,2),mask_min_pixel_percentage=0.2):
		if mask_min_pixel_percentage=="any":
			return patches[np.any(mask==domain,axis=axis),::]
		else:
			pixel_limit=int(patches.shape[1]*patches.shape[2]*mask_min_pixel_percentage)
			print("Max pixel {}, pixel limit {}".format(patches.shape[1]*patches.shape[2],pixel_limit))
			return patches[np.count_nonzero(mask==domain,axis=axis)>pixel_limit,::]
def patches_from_subset(subset,data,window_shape,patches_step, \
	mask_min_pixel_percentage, wildfire_min_pixel_percentage=-1,
	mask_value=1):
	print("Starting patch extraction...")
	subset['im'],_=view_as_windows_flat(data['im'],window_shape,step=(patches_step,patches_step,data['im'].shape[2]))
	subset['mask'],_=view_as_windows_flat(data['mask'],(window_len,window_len),step=(patches_step,patches_step))
	subset['label'],_=view_as_windows_flat(data['label'],(window_len,window_len),step=(patches_step,patches_step))
	deb.prints(np.unique(subset['mask'],return_counts=True))
	deb.prints(np.unique(data['mask'],return_counts=True))
	
	deb.prints(subset['im'].shape)
	deb.prints(mask_min_pixel_percentage)
	subset['im']=patches_from_domain_gather(subset['im'],subset['mask'], \
		mask_value,mask_min_pixel_percentage=mask_min_pixel_percentage)
	subset['label']=patches_from_domain_gather(subset['label'],subset['mask'], \
		mask_value,mask_min_pixel_percentage=mask_min_pixel_percentage)
	deb.prints(subset['im'].shape)

	condition = (wildfire_min_pixel_percentage=="any") if \
		isinstance(wildfire_min_pixel_percentage,str) else \
		(wildfire_min_pixel_percentage>0)

	if condition:
		print("Taking only wildfire patches, {}".format(wildfire_min_pixel_percentage))
		subset['im']=patches_from_domain_gather(subset['im'], \
			subset['label'],2,mask_min_pixel_percentage=wildfire_min_pixel_percentage)
		subset['label']=patches_from_domain_gather(subset['label'], \
			subset['label'],2,mask_min_pixel_percentage=wildfire_min_pixel_percentage)
	deb.prints(subset['im'].shape)
	return subset


from  skimage.util import view_as_windows

window_len=a.window_len
channel_n=a.channel_n
#patches_step=int(window_len/3)
#patches_step=int(window_len)
#patches_step=a.train_step
deb.prints(a.train_step)
deb.prints(a.test_step)

window_shape=(window_len,window_len,channel_n)

deb.prints(np.unique(data['train']['label'],return_counts=True))
deb.prints(np.unique(data['test']['label'],return_counts=True))

patches={'train':{},'test':{},'val':{}}
print("Train patches...")
patches['train']=patches_from_subset(patches['train'],data['train'], \
	window_shape,int(a.train_step),mask_min_pixel_percentage=0.1, \
	wildfire_min_pixel_percentage=a.wildfire_min_pixel_percentage,
	mask_value=1)

print("Extracting test patches...")
patches['test']=patches_from_subset(patches['test'],data['test'], \
	window_shape,int(a.test_step),mask_min_pixel_percentage="any",
	mask_value=2) #\
	#wildfire_min_pixel_percentage=a.wildfire_min_pixel_percentage)
if a.validating==True:
	print("Extracting val patches...")
	patches['val']=patches_from_subset(patches['val'],data['val'], \
		window_shape,int(a.train_step),mask_min_pixel_percentage="any",
		mask_value=3)

deb.prints(patches['train']['im'].shape)
deb.prints(patches['test']['im'].shape)
if a.validating==True:
	deb.prints(patches['val']['im'].shape)

folder="compact/"+dataset+"/"
pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

deb.prints(np.unique(patches['test']['mask'],
	return_counts=True))
np.save(folder+"train_im.npy",patches['train']['im'])
np.save(folder+"train_label.npy",patches['train']['label'])
np.save(folder+"test_im.npy",patches['test']['im'])
np.save(folder+"test_label.npy",patches['test']['label'])
if a.validating==True:
	np.save(folder+"val_im.npy",patches['val']['im'])
	np.save(folder+"val_label.npy",patches['val']['label'])

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
	patches_store(patches['im'],a.output_folder+dataset+"/im/")

	patches_store(patches['mask'],a.output_folder+dataset+"/mask/")

	patches_store(patches['label'],a.output_folder+dataset+"/label/")
