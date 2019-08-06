import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import random
import keras
from keras.engine.input_layer import Input
import keras
import numpy as np
import h5py
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
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
from keras.preprocessing.image import ImageDataGenerator

#load the dataset
#processed
sites = np.load("/sdata/neuroimaging/processed/numpy/2019-07-22-processed_filteredage_site.npy",allow_pickle=True)
print(sites.shape)
imgsP = np.load("/sdata/neuroimaging/processed/numpy/2019-07-22-processed_filtered_img.npy",allow_pickle=True)

## image augmentation
def translateit(image, isseg=False):
    if random.randint(0,1) == 0:
        offset = (random.randint(-10,10), random.randint(-10,10),1)
        order = 0 if isseg == True else 5
        return scipy.ndimage.interpolation.shift(image, offset, order=order, mode='nearest')
    else:
        return image
    
image = x_train[2]
augImg = translateit(image, isseg=True)
#plt.imshow(augImg)
v = np.array(augImg == image)
d = np.array(augImg != image)
print(v.all())
if v.all():
    print("same")
else:
    print("different")
#image_generator function notice it only returns batch_x
def image_generator(files):
    while True:
        augImgs = []
        labels = []
        for i in range(len(files)):
            augImg = translateit(files[i], isseg=False)
            augImgs.append(augImg)
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(augImgs)
        #batch_y = np.array(labels)
        return batch_x

"""
Finally the moment you've all been waiting for
"""

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()


# Generators
training_generator = image_generator(x_train)
test_generator = image_generator(x_test)
print("made it past this")
print(training_generator.shape)

# applying transformation to image
train_gen = ImageDataGenerator()
test_gen = ImageDataGenerator()
training_set= train_gen.flow(training_generator, y_train, batch_size=64)
test_set= test_gen.flow(test_generator, y_test, batch_size=64)

#set up the callbacks
es = keras.callbacks.EarlyStopping(monitor='val_loss',
                    min_delta=0,
                    patience=5,
                    verbose=0, mode='auto')

cp = keras.callbacks.ModelCheckpoint(filepath='/wdata/rotating_students/gramos/hackathon/image_augmentation/aug_best_model',
                                verbose=1,
                                save_best_only=True)

# Train model on dataset
model.fit_generator(training_set, 
                         steps_per_epoch=60000//64, 
                         validation_data= test_set, 
                         validation_steps=10000//64, 
                         epochs=5,
                         callbacks=[es,cp])