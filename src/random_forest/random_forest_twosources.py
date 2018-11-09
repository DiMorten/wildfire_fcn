

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

def balance_data(data, labels, samples_per_class):

	shape = data.shape
	if len(shape) > 2:
		data = data.reshape(shape[0], shape[1]*shape[2]*shape[3])
	classes = np.unique(labels)
	print(classes)
	num_total_samples = len(classes)*samples_per_class
	out_labels = np.zeros((num_total_samples), dtype='float32')
	out_data = np.zeros((num_total_samples, data.shape[1]), dtype='float32')

	k = 0
	for clss in classes:
		clss_labels = labels[labels == clss]
		clss_data = data[labels == clss]
		num_samples = len(clss_labels)
		if num_samples > samples_per_class:
			# Choose samples randomly
			index = range(len(clss_labels))
			index = np.random.choice(index, samples_per_class, replace=False)
			out_labels[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_labels[index]
			out_data[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_data[index]

		else:
			# do oversampling
			index = range(len(clss_labels))
			index = np.random.choice(index, samples_per_class, replace=True)
			out_labels[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_labels[index]
			out_data[k*samples_per_class:k*samples_per_class + samples_per_class] = clss_data[index]
		k += 1
	# Permute samples randomly
	idx = np.random.permutation(out_data.shape[0])
	out_data = out_data[idx]
	out_labels = out_labels[idx]

	if len(shape) > 2:
		out_data = out_data.reshape(out_data.shape[0], shape[1], shape[2], shape[3])

	return out_data, out_labels

def stack_images(images_list, seq):

	#print(images_list[seq[0]-1])
	h5file = h5py.File(images_list[seq[0]-1])
	fileHeader = h5file['features']
	img = np.float32(fileHeader[:]).T
	h5file.close()
	print(img.shape)
	rows, cols = img.shape
	stack = np.zeros((rows, len(seq) * cols), dtype='float32')
	#stack[:, 0:cols] = img
	img = []
	cont = 0
	for i in seq:
		print(images_list[i])
		h5file = h5py.File(images_list[i])
		fileHeader = h5file['features']
		print(np.min(fileHeader),np.max(fileHeader))
		stack[:, cols*cont:cols*cont+cols] = np.float32(fileHeader[:]).T
		h5file.close()
		cont += 1
	return stack

def load_image(DIM, patch):
	# Read Mask Image
	print(patch[0])
	gdal_header = gdal.Open(patch[0])
	img = gdal_header.ReadAsArray()
	img = img.reshape(DIM[0] * DIM[1], order='F')
	return img
def load_image2(patch):
	# Read Mask Image
	print(patch[0])
	gdal_header = gdal.Open(patch[0])
	img = gdal_header.ReadAsArray()
	return img

def fortran_flatten(img):
	DIM = img.shape
	return img.reshape(DIM[0] * DIM[1], order='F')

def im_load(path):


	# Read image
	im = gdal.Open(path['raster'])
	im = np.array(im.ReadAsArray())
	im = np.transpose(im, (1, 2, 0))
	if dataset=='para':
		im = im[0:-1,0:-1,:] # Eliminate non used pixels
	elif dataset=='acre':
		im = im[0:-1,:,:]
	im = im.astype(np.float32)
	deb.prints(im.shape)
	deb.prints(im.dtype)

	im = im.reshape(im.shape[0]*im.shape[1],-1)
	deb.prints(im.shape)
	return im

def statistics_print(label,mask,label_train):


	unique,count=np.unique(label,return_counts=True)
	print(unique,count)

	unique,count=np.unique(mask,return_counts=True)
	print("Mask",unique,count)


	unique,count=np.unique(label_train,return_counts=True)
	print("Label train unique",unique,count)


	print("Mask count total",np.count_nonzero(mask))
	print("Mask count train",np.count_nonzero(mask[mask==1]))
	print("Mask count test",np.count_nonzero(mask[mask==2]))
	print("Label count total",np.count_nonzero(label))
	print("Label",label.shape)
	print("Mask",mask.shape)

def path_configure(dataset):

	path={}
	if dataset=='para':
		path['data']='../../data/AP1_Para/'
		path['raster']=path['data']+'L8_224-66_ROI_clip.tif'
		path['label']=path['data']+'labels.tif'
		path['bounding_box']=path['data']+'bounding_box_pa_clip.tif'
	elif dataset=='acre':
		path['data']='../../data/AP2_Acre/'
		path['raster']=path['data']+'L8_002-67_ROI.tif'
		path['label']=path['data']+'labels.tif'
		path['bounding_box']=path['data']+'bounding_box_clip.tif'
	path['train_test_mask']=path['data']+'TrainTestMask.png'
	return path	

def mask_label_load(path):

	mask=cv2.imread(path['train_test_mask'],0).astype(np.uint8)
	label=cv2.imread(path['label'],-1).astype(np.uint8)
	label[label==2]=1 # Only use 2 classes
	label=label+1 # 0 is for background

	bounding_box=cv2.imread(path['bounding_box'],-1).astype(np.uint8)
	mask[mask==255]=1
	mask=mask+1
	mask[bounding_box==0]=0 # Background. No data
	label[bounding_box==0]=0 # Background. No data
	mask=mask.reshape(-1)
	label=label.reshape(-1)
	# Not quite necessary to do this but more informative

	return mask,label

#================== DEFINE FILENAMES =======================
load_other_model=True

#dataset='acre'
dataset='para'
path=path_configure(dataset)
mask,label=mask_label_load(path)

#============= LOAD GLCM DATA AND TRAIN=======================


# ================== STACK IMAGES ============================  
im=im_load(path)
deb.prints(mask.shape)
# ================== MASK THE IMAGES ===================
features_train=im[mask==2]
features_test=im[mask==1]
del im

label_train=label[mask==2]
label_test=label[mask==1]

label=label[mask!=0]
mask=mask[mask!=0]

print(features_train.shape)
print(features_test.shape)


#================= PRINT STATISTICS===============
statistics_print(label,mask,label_train)



# ================== Normalize

scaler = pp.StandardScaler().fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

samples_per_class=300000


# ================== DATA BALANCE ============================


features_train, label_train=balance_data(features_train, label_train, 
	samples_per_class=samples_per_class)

print(features_train.shape) #(600000, 6)
print(label_train.shape) #(600000)



#np.save('features_train.npy',features_train)
#np.save('features_test.npy',features_test)
#np.save('labels_train.npy',label_train)
#np.save('labels_test.npy',label_test)



#================== START TRAINING=======================


n_trees=250
max_depth=25


# SKLEARN  Classifier
#start_time = timeit.default_timer()
clf = RandomForestClassifier(n_estimators=n_trees,
							 max_depth=max_depth,
							 n_jobs=-1)
#start_time = time.time()

if load_other_model==False:
	print('Start training...............')
	clf = clf.fit(features_train, label_train)
	joblib.dump(clf, 'trained_classifier.joblib') 
	print('Training finished, time of executuion ')

if load_other_model==True:
	print("Loading other model...")
	clf = joblib.load('results/acre/trained_classifier.joblib') 

# predict
#start_time = time.time()
print('Start testing...............')
predict_batch = 200000
predictions = np.zeros((np.shape(features_test)[0]))
for i in range(0, np.shape(features_test)[0], predict_batch):
	predictions[i:i+predict_batch] = clf.predict(features_test[
		i:i+predict_batch])

predictions_prob = np.zeros((np.shape(features_test)[0],len(np.unique(label_test))))
for i in range(0, np.shape(features_test)[0], predict_batch):
	predictions_prob[i:i+predict_batch] = clf.predict_proba(features_test[
		i:i+predict_batch])
np.save('predictions.npy',predictions)

np.save('predictions_prob.npy',predictions_prob)
predictions=predictions.astype(np.uint8)
#finish = time.time()
#test_time = (finish - start_time)
#print('Test finished, time of executuion ', test_time/60)
#else:
	#predictions=np.load('predictions.npy').astype(np.uint8)
	#predictions=np.load('seq1/predictions_300k.npy').astype(np.uint8)
#    predictions=np.load('seq2/predictions_seq2_300k.npy').astype(np.uint8)
	
	
print("predictions",predictions.shape,np.unique(predictions),predictions.dtype)
print("label_test",label_test.shape,np.unique(label_test),label_test.dtype)
predictions=predictions.astype(np.uint8)

metrics={}
metrics['f1_score']=f1_score(label_test,predictions,average='macro')
metrics['f1_score_weighted']=f1_score(label_test,predictions,average='weighted')
		
metrics['overall_acc']=accuracy_score(label_test,predictions)
confusion_matrix_=confusion_matrix(label_test,predictions)
metrics['per_class_acc']=(confusion_matrix_.astype('float') / confusion_matrix_.sum(axis=1)[:, np.newaxis]).diagonal()
		
metrics['average_acc']=np.average(metrics['per_class_acc'][~np.isnan(metrics['per_class_acc'])])
print(metrics)
print(confusion_matrix_)







## 
#  [ 1  2  6  7  8  9 10 11] [  35399   27469   80238  245754   83364 2329914     308   89488]
# For seq2.