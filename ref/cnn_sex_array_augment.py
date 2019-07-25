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

## image augmentation
def translateit(image, isseg=False):
    if random.randint(0,1) == 0:
        offset = (random.randint(-10,10), random.randint(-10,10), random.randint(-10,10), 0)
        order = 0 if isseg == True else 5
        return scipy.ndimage.interpolation.shift(image, offset, order=order, mode='nearest')
    else:
        return image


img_trans = scipy.ndimage.interpolation.shift(img, (2,10,0,0) )
plt.imshow(img[img.shape[0] // 2,:,:,0])
plt.savefig('/Dedicated/jmichaelson-wdata/tkoomar/img.png')
plt.imshow(img_trans[img_trans.shape[0] // 2,:,:,0])
plt.savefig('/Dedicated/jmichaelson-wdata/tkoomar/img_trans.png')


# model callbacks
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                    min_delta=0,
                    patience=5,
                    verbose=0, mode='auto')
cp = keras.callbacks.ModelCheckpoint(filepath='/Dedicated/jmichaelson-wdata/lbrueggeman/brain_mri_domain_adaptation/models/domain_adaptation/cnn-set{}'.format(batch_num),
                                verbose=1,
                                save_best_only=True)

# train model
model = con_net()
model.fit(x=x[train_ind],
          y=age[train_ind],
          validation_data=(x[val_ind], age[val_ind]),
          batch_size=32,
          epochs=100,
          verbose=1,
          shuffle=True,
          callbacks=[es,cp])
