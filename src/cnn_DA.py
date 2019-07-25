# array job batch index
import sys
batch_num = int(sys.argv[1])

# function import
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

# import domain adaptation functions
# source: https://github.com/michetonu/gradient_reversal_keras_tf
import os
os.chdir('/Dedicated/jmichaelson-wdata/lbrueggeman/brain_mri_domain_adaptation/src')
import DA_functions

# load data for a normal fit procedure
x = np.load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filteredage_img.npy")
x = np.expand_dims(x, axis=4)
age = np.load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filteredage_age_recode.npy")

# load in site 
study = np.load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filteredage_study.npy")
site = np.load("/Dedicated/jmichaelson-sdata/neuroimaging/processed/numpy/2019-07-22-processed_filteredage_site.npy")
study_site = np.core.defchararray.add(study.astype(np.str),site.astype(np.str))
from sklearn.preprocessing import LabelEncoder
sitesT = np.reshape(study_site, (-1,1))
LE = LabelEncoder()
LE_site = LE.fit_transform(sitesT)
unique = np.unique(study_site)
nb_classes = unique.shape[0]
targets = np.array(LE_site).reshape(-1)
oh_site = np.eye(nb_classes)[targets]
#np.where(oh_site[1]) test
#unique[37] test

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
## head 1 - age prediction
    head1 = Dense(1, activation='linear', bias_initializer=keras.initializers.Constant(value=20), name='head1')(flatten1)
## head 2 - domain adaptation
    Flip = DA_functions.GradientReversal(1)
    head2_flip = Flip(flatten1)
    head2 = Dense(93, activation='sigmoid', bias_initializer=keras.initializers.Constant(value=0), name='head2')(head2_flip)
## define the model with input layer and output layer
    adam = keras.optimizers.Adam(lr=1e-4)
    model = Model(inputs=input_layer, outputs=[head1, head2])
    loss_funcs = {"head1":"mean_squared_error", "head2":"categorical_crossentropy"}
    loss_weights = {"head1":1.0, "head2":1.0}
    metrics = {"head1":["mean_squared_error","mean_absolute_error"], "head2":["categorical_crossentropy","accuracy"]}
    model.compile(loss=loss_funcs,
        loss_weights=loss_weights,
        metrics=metrics,
        optimizer=adam)     
    return model

# model callbacks
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                    min_delta=0,
                    patience=5,
                    verbose=0, mode='auto')
cp = keras.callbacks.ModelCheckpoint(filepath='/Dedicated/jmichaelson-wdata/lbrueggeman/brain_mri_domain_adaptation/models/domain_adaptation/cnn-set{}'.format(batch_num),
                                verbose=1,
                                save_best_only=True)

y_trains = {"head1":age[train_ind],
            "head2":oh_site[train_ind]}
y_valids = {"head1":age[val_ind],
            "head2":oh_site[val_ind]}

# train model
model = con_net()
model.fit(x=x[train_ind],
          y=y_trains,
          validation_data=(x[val_ind], y_valids),
          batch_size=32,
          epochs=100,
          verbose=1,
          shuffle=True,
          callbacks=[es,cp])


