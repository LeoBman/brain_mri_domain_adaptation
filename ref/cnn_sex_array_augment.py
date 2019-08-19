# array job batch index
import sys
batch_num = int(sys.argv[1])

import scipy
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# function imports
import numpy as np
import keras
from keras.utils.io_utils import HDF5Matrix
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.layers import Conv3D, MaxPooling3D, Dropout,Conv1D, MaxPooling1D,Conv2D, MaxPooling2D,MaxPool3D
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dropout, Input, BatchNormalization
from keras.activations import relu
from keras.layers import Dense, Activation
from keras.constraints import max_norm
import h5py
from keras.preprocessing.image import ImageDataGenerator

# load data for a normal fit procedure
x = np.load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filteredage_img.npy")
x = np.expand_dims(x, axis=4)
age = np.load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filteredage_age_recode.npy")

# filter age
age_filt = np.invert(np.isnan(age))
age_filt1 = np.invert(age<0)
age_filt = age_filt & age_filt1
keep_ind = np.where(age_filt)[0]

# load in cv folds and get subset within keep_ind (no NAs, no negative values)
folds = np.load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-folds.npy")
for i in range(5):
    for j in range(3):
        folds[i,j] = folds[i,j][np.isin(folds[i,j], keep_ind)]

# set train and validation indices based on sys arg (batch_num above)
folds_bool = np.isin(range(1,6,1), batch_num, invert=True)
train_ind = folds[folds_bool,0]
train_ind = np.hstack(train_ind)
val_ind = folds[folds_bool,1]
val_ind = np.hstack(val_ind)


## image augmentation
## adapted from https://mlnotebook.github.io/post/dataaug/
def translateit(image, isseg=False):
    if random.randint(0,1) == 0:
        offset = (random.randint(-20,20), random.randint(-20,20), random.randint(-20,20), 0)
        order = 0 if isseg == True else 5
        return scipy.ndimage.interpolation.shift(image, offset, order=order, mode='nearest')
    else:
        return image

def scaleit(image, isseg=False):
    factor = random.randint(72,76) / random.randint(72,76)
    if random.randint(0,1) == 1:
      factor = 1
    order = 0 if isseg == True else 3
    height, width, depth = image.shape[0:3]
    zheight             = int(np.round(factor * height))
    zwidth              = int(np.round(factor * width))
    zdepth              = depth
    if factor < 1.0:
        image = image[:,:,:,0]
        newimg  = np.zeros_like(image)
        row     = (height - zheight) // 2
        col     = (width - zwidth) // 2
        layer   = (depth - zdepth) // 2
        newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] =  scipy.ndimage.interpolation.zoom(image, (float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]
        return newimg[..., np.newaxis]
    elif factor > 1.0:
        image = image[:,:,:,0]
        row     = (zheight - height) // 2
        col     = (zwidth - width) // 2
        layer   = (zdepth - depth) // 2
        newimg =  scipy.ndimage.interpolation.zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0), order=order, mode='nearest')  
        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]
        return newimg[..., np.newaxis]
    else:
        return image

def rotateit(image, isseg=False):
    order = 0 if isseg == True else 5
    if random.randint(0,1) == 0:
        theta = random.randint(-15,15)  
        return scipy.ndimage.rotate(image, float(theta), reshape=False, order=order, mode='nearest')
    else:
        return image

def flipit(image):
    if random.randint(0,2) == 0:
        image = np.fliplr(image)
    if random.randint(0,2) == 0:
        image = np.flipud(image)
    return image

# img = x[0]
# img_trans = flipit(img)
# plt.imshow(img[img.shape[0] // 2,:,:,0])
# plt.savefig('/Dedicated/jmichaelson-wdata/tkoomar/img.png')
# plt.imshow(img_trans[img_trans.shape[0] // 2,:,:,0])
# plt.savefig('/Dedicated/jmichaelson-wdata/tkoomar/img_trans.png')

# model definition
def con_net():
## input layer
    keras.backend.clear_session()
    input_layer = Input((91,109,91,1))
## conv 1
    conv1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last')(input_layer)
    pool1 = MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv1)
    batch1 = BatchNormalization()(pool1)
## conv 2
    conv2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(batch1)
    pool2 = MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv2)
    batch2 = BatchNormalization()(pool2)
## conv 3
    conv3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(batch2)
    conv4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv3)
    pool3 = MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv4)
    batch3 = BatchNormalization()(pool3)
## conv 4
    conv5 = Conv3D(filters=128, kernel_size=(2, 2, 2), activation='relu')(batch3)
    conv6 = Conv3D(filters=256, kernel_size=(2, 2, 2), activation='relu')(conv5)
    pool4 = MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv6)
    flatten1 = Flatten()(pool4)
    output_layer = Dense(1, activation='linear', bias_initializer=keras.initializers.Constant(value=20))(flatten1)
    ## define the model with input layer and output layer
    adam = keras.optimizers.Adam(lr=1e-4)
    model = Model(inputs=input_layer, outputs=output_layer) 
    model.compile(loss='mean_squared_error', metrics=['mean_absolute_error'], optimizer=adam)     
    return model

#image_generator function notice it only returns batch_x
def image_generator(x, y, indicies, scale = True, translate = True, flip = True, rotate = True, batch_size = 32):
    batch = np.random.choice(indicies, batch_size)
    while True:
        augImgs = []
        labels =  y[batch]
        for i in range(len(batch)):
            j = batch[i]
            augImg = x[j]
            if translate : 
                augImg = translateit(augImg)
            if scale :
                augImg = scaleit(augImg)
            if flip : 
                augImg = flipit(augImg)
            if rotate :
                augImg = rotateit(augImg)
            augImgs.append(augImg)
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(augImgs)
        batch_y = np.array(labels)
        yield( batch_x, batch_y )


# applying transformation to image
#train_gen = ImageDataGenerator()
#val_gen = ImageDataGenerator()
#training_set = train_gen.flow(image_generator(x[train_ind]), age[train_ind], batch_size=4)
#validation_set= test_gen.flow(image_generator(x[val_ind]), x[val_ind], batch_size=4)

# model callbacks
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                    min_delta=0,
                    patience=10,
                    verbose=0, mode='auto')

cp = keras.callbacks.ModelCheckpoint(filepath='../models/augment/cnn-set{}'.format(batch_num), 
    verbose=1,
    save_best_only=True)

# train model
model = con_net()
batch_size = 32

model.fit_generator(image_generator(x, age, train_ind, scale = False, batch_size = batch_size),
        steps_per_epoch= 10, 
        validation_data = image_generator(x, age, val_ind, scale = False, translate = False, rotate = False, flip = False, batch_size = batch_size), 
        validation_steps= val_ind.size // batch_size, 
        epochs=100, 
        verbose = 1, 
        callbacks=[es,cp])
