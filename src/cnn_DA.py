# array job batch index
import sys
batch_num = int(sys.argv[1])

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

# import data
x = np.load('/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filtered_
img.npy')
x = np.expand_dims(x, axis=4)
sex = np.load('/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filtered_sex.npy')

# load index splits for 5 fold CV
cv = np.load('/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-folds.npy')


# model definition
def con_net():
## input layer
    keras.backend.clear_session()
    input_layer = Input((128,128,128,1))
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
    output_layer = Dense(1, activation='sigmoid', bias_initializer=keras.initializers.Constant(value=1))(flatten1)
    ## define the model with input layer and output layer
    adam = keras.optimizers.Adam(lr=1e-5)
    model = Model(inputs=input_layer, outputs=output_layer) 
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=adam)     
    return model

# model callbacks
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                    min_delta=0,
                    patience=5,
                    verbose=0, mode='auto')
cp = keras.callbacks.ModelCheckpoint(filepath='/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/trained_models/2019-06-24-cnn-set{}'.format(batch_num),
                                verbose=1,
                                save_best_only=True)

# train model
model = con_net()
model.fit(x=x[np.isin(range(21390), ind_split[batch_num], invert=True)],
          y=sex[np.isin(range(21390), ind_split[batch_num], invert=True)],
          validation_split=0.15,
          batch_size=32,
          epochs=50,
          verbose=1,
         shuffle=False,
         callbacks=[es, cp])































