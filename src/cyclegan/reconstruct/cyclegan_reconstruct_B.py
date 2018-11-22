import os
import cv2
import deb
from osgeo import gdal
import numpy as np
from sklearn.externals import joblib

os.environ['KERAS_BACKEND']='tensorflow' # can choose theano, tensorflow, cntk
os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_run,dnn.library_path=/usr/lib'
#os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,optimizer=fast_compile,dnn.library_path=/usr/lib'



import keras.backend as K
if os.environ['KERAS_BACKEND'] =='theano':
    channel_axis=1
    K.set_image_data_format('channels_first')
    channel_first = True
else:
    K.set_image_data_format('channels_last')
    channel_axis=-1
    channel_first = False



from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
# ==== Choose if importing weights

weight_load=True
weights_path="../acre_to_para/"
#weights_path="first_model/"

# Weights initializations
# bias are initailized as 0
def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization


# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = conv_init, *a, **k)
def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)
def BASIC_D(nc_in, ndf, max_layers=3, use_sigmoid=True):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """    
    if channel_first:
        input_a =  Input(shape=(nc_in, None, None))
    else:
        input_a = Input(shape=(None, None, nc_in)) # Here I might put 128
    _ = input_a
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First') (_)
    _ = LeakyReLU(alpha=0.2)(_)
    
    for layer in range(1, max_layers):        
        out_feat = ndf * min(2**layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same", 
                   use_bias=False, name = 'pyramid.{0}'.format(layer)             
                        ) (_)
        _ = batchnorm()(_, training=1)        
        _ = LeakyReLU(alpha=0.2)(_)
    
    out_feat = ndf*min(2**max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last') (_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)
    
    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1), 
               activation = "sigmoid" if use_sigmoid else None) (_)    
    return Model(inputs=[input_a], outputs=_)



def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):    
    max_nf = 8*ngf    
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s)) (x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=channel_axis)([x, x2])            
        x = Activation("relu")(x)
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init,          
                            name = 'convt.{0}'.format(s))(x)        
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <=8:
            x = Dropout(0.5)(x, training=1)
        return x
    
    s = isize if fixed_input_size else None
    if channel_first:
        _ = inputs = Input(shape=(nc_in, s, s))
    else:
        _ = inputs = Input(shape=(s, s, nc_in))        
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    #_ = Activation('tanh')(_)
    #_ = LeakyReLU(alpha=0.2)(_)
    #_ = Activation('linear')(_)
    return Model(inputs=inputs, outputs=[_])



nc_in = 6
nc_out = 6
ngf = 64
ndf = 64
use_lsgan = True
λ = 10 if use_lsgan else 100

loadSize = 143
imageSize = 128
batchSize = 1
lrD = 2e-4
lrG = 2e-4



netDA = BASIC_D(nc_in, ndf, use_sigmoid = not use_lsgan)
netDB = BASIC_D(nc_out, ndf, use_sigmoid = not use_lsgan)
netDA.summary()



from keras.utils.vis_utils import model_to_dot


netGB = UNET_G(imageSize, nc_in, nc_out, ngf)
netGA = UNET_G(imageSize, nc_out, nc_in, ngf)
#SVG(model_to_dot(netG, show_shapes=True).create(prog='dot', format='svg'))
netGA.summary()


if weight_load==True:

    netDA.load_weights(weights_path+"netDA.h5")
    netDB.load_weights(weights_path+"netDB.h5")
    netGA.load_weights(weights_path+"netGA.h5")
    netGB.load_weights(weights_path+"netGB.h5")


from keras.optimizers import RMSprop, SGD, Adam



if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target)) # Cross entropy

def cycle_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    fn_generate = K.function([real_input], [fake_output, rec_input])
    return real_input, fake_output, rec_input, fn_generate

real_A, fake_B, rec_A, cycleA_generate = cycle_variables(netGB, netGA)
real_B, fake_A, rec_B, cycleB_generate = cycle_variables(netGA, netGB)


# =========== Reconstruct
def G(fn_generate, X):
        r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
        return r.swapaxes(0,1)[:,:,0]   
def path_configure(dataset,source='tiff',train_test_mask='TrainTestMask.png'):

    path={}
    if dataset=='para':
        path['data']='../../../data/AP1_Para/'
        path['raster']=path['data']+'L8_224-66_ROI_clip.tif'
        path['label']=path['data']+'labels.tif'
        path['bounding_box']=path['data']+'bounding_box_pa_clip.tif'
    elif dataset=='acre':
        path['data']='../../../data/AP2_Acre/'
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

def im_load(path,dataset,source='tiff'):
    # Read image
    if source=='tiff':
        im = gdal.Open(path['raster'])
        im = np.array(im.ReadAsArray())
        im = np.transpose(im, (1, 2, 0))
        im = im.astype(np.float32)
    elif source=='matlab':
        #print(images_list[seq[0]-1])
        h5file = h5py.File(path['raster'],'r')
        fileHeader = h5file['dataset']
        print(fileHeader)
        im = np.float32(fileHeader).T
        h5file.close()
        print(im.shape)
    if dataset=='para':
        im = im[0:-1,0:-1,:] # Eliminate non used pixels
    elif dataset=='acre':
        im = im[0:-1,:,:]
    
    deb.prints(im.shape)
    deb.prints(im.dtype)

    
    return im
def stats_print(x):
    print(np.min(x),np.max(x),np.average(x),x.dtype)

def im_reconstruct(img,cycleB_generate,window,overlap):
    out=np.zeros_like(img)

    h, w, channels = img.shape
        

    gridx = range(0, w - window, window - overlap)
    gridx = np.hstack((gridx, w - window))

    gridy = range(0, h - window, window - overlap)
    gridy = np.hstack((gridy, h - window))
    deb.prints(gridx.shape)
    deb.prints(gridy.shape)
        
    counter=0
    
    # Load scalers
    scalerA = joblib.load('../../patch_extract2/scaler_acre.joblib') 
    scalerB = joblib.load('../../patch_extract2/scaler_para.joblib') 

    deb.prints(im.shape)
    stats_print(im[:,:,0:3])
    cv2.imwrite("in.png",img[:,:,0:3]/4)
    # Normalize img which is B
    img = img.reshape(h*w,-1)
    img = scalerB.transform(img)
    img = img.reshape(h,w,channels)
    #======================== START IMG LOOP ==================================#
    for i in range(len(gridx)):
        for j in range(len(gridy)):
            counter=counter+1
            if counter % 10000 == 0:
                deb.prints(counter,fname)
            xx = gridx[i]
            yy = gridy[j]
            #patch_clouds=Bclouds[yy: yy + window, xx: xx + window]
            B = img[yy: yy + window, xx: xx + window,:].copy()
            A = G(cycleB_generate,np.expand_dims(B,axis=0))
            A_ = A[0][0]
            # Unnormalize

            A_ = np.reshape(A_,(window*window,-1))
            A_ = scalerA.inverse_transform(A_)
            A_ = np.reshape(A_,(window,window,channels))

            out[yy: yy + window, xx: xx + window,:]=A_.copy()
    stats_print(out[:,:,0:3])
    np.save("out.npy",out)
    cv2.imwrite("out.png",out[:,:,0:3]/4)

dataset='para'
path=path_configure(dataset)
im=im_load(path,dataset)
deb.prints(im.shape)
window=128
overlap=0
im_reconstruct(im,cycleB_generate,window,overlap)