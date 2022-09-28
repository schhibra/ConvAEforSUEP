##########################################
import setGPU

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
import glob

import numpy      as np
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
####config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
##########################################

##########################################
# keras imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Dropout, Flatten, Reshape, Lambda, Layer, LeakyReLU
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
##########################################

########################################## 
from keras.preprocessing.image import ImageDataGenerator

INPUT_DIM = (128,128,3) # Image dimension
parts = 256
Z_DIM = 200 # Dimension of the latent vector (z)

data_flow = ImageDataGenerator(rescale=1./255).flow_from_directory('./img_align_celeba/', 
                                                                   target_size = INPUT_DIM[:2],
                                                                   batch_size = parts,
                                                                   shuffle = True,
                                                                   class_mode = 'input',
                                                                   subset = 'training')
########################################## 

##########################################
img_rows = 128
img_cols = 128
img_chns = 3

############ENCODER
inputImage = Input(shape=(img_rows, img_cols, img_chns))
x1 = Conv2D(32, (3,3), padding="same")(inputImage)
x2 = BatchNormalization()(x1)
x3 = Activation('elu')(x2)
x4 = AveragePooling2D((2,2), padding="same")(x3)
x5 = Conv2D(64, (3,3), padding="same")(x4)
x6 = BatchNormalization()(x5)
x7 = Activation('elu')(x6)
x8 = AveragePooling2D((2,2), padding="same")(x7)
x9 = Conv2D(64, (3,3), padding="same")(x8)
x10 = BatchNormalization()(x9)
x11 = Activation('elu')(x10)
x12 = AveragePooling2D((2,2), padding="same")(x11)
x13 = Conv2D(64, (3,3), padding="same")(x12)
x14 = BatchNormalization()(x13)
x15 = Activation('elu')(x14)
x16 = AveragePooling2D((2,2), padding="same")(x15)
x17 = Conv2D(64, (2,2), padding="same")(x16)
x18 = BatchNormalization()(x17)
x19 = Activation('elu')(x18)
x20 = AveragePooling2D((2,2), padding="same")(x19)

x21 = Conv2D(64, (2,2), padding="same")(x20)
x22 = BatchNormalization()(x21)
encoder_output = Activation('elu')(x22)

############DECODER
x23 = Conv2DTranspose(64, (2,2), strides=(2, 2), padding="same")(encoder_output)
x24 = BatchNormalization()(x23)
x25 = Activation('elu')(x24)
x26 = Conv2DTranspose(64, (2,2), strides=(2, 2), padding="same")(x25)
x27 = BatchNormalization()(x26)
x28 = Activation('elu')(x27)
x29 = Conv2DTranspose(64, (2,2), strides=(2, 2), padding="same")(x28)
x30 = BatchNormalization()(x29)
x31 = Activation('elu')(x30)
x32 = Conv2DTranspose(64, (2,2), strides=(2, 2), padding="same")(x31)
x33 = BatchNormalization()(x32)
x34 = Activation('elu')(x33)
x35 = Conv2DTranspose(img_chns, (2,2), strides=(2, 2), padding="same")(x34)
x36 = BatchNormalization()(x35)
output = Activation('elu')(x36)

model   = Model(inputs=inputImage, outputs=output)
encoder = Model(inputs=inputImage, outputs=encoder_output)
m1 = Model(inputs=inputImage, outputs=x1)
m2 = Model(inputs=inputImage, outputs=x2)
m3 = Model(inputs=inputImage, outputs=x3)
m4 = Model(inputs=inputImage, outputs=x4)
m5 = Model(inputs=inputImage, outputs=x5)
m6 = Model(inputs=inputImage, outputs=x6)
m7 = Model(inputs=inputImage, outputs=x7)
m8 = Model(inputs=inputImage, outputs=x8)
m9 = Model(inputs=inputImage, outputs=x9)
m10 = Model(inputs=inputImage, outputs=x10)
m11 = Model(inputs=inputImage, outputs=x11)
m12 = Model(inputs=inputImage, outputs=x12)
m13 = Model(inputs=inputImage, outputs=x13)
m14 = Model(inputs=inputImage, outputs=x14)
m15 = Model(inputs=inputImage, outputs=x15)
m16 = Model(inputs=inputImage, outputs=x16)
m17 = Model(inputs=inputImage, outputs=x17)
m18 = Model(inputs=inputImage, outputs=x18)
m19 = Model(inputs=inputImage, outputs=x19)
m20 = Model(inputs=inputImage, outputs=x20)
m21 = Model(inputs=inputImage, outputs=x21)
m22 = Model(inputs=inputImage, outputs=x22)
m23 = Model(inputs=inputImage, outputs=x23)
m24 = Model(inputs=inputImage, outputs=x24)
m25 = Model(inputs=inputImage, outputs=x25)
m26 = Model(inputs=inputImage, outputs=x26)
m27 = Model(inputs=inputImage, outputs=x27)
m28 = Model(inputs=inputImage, outputs=x28)
m29 = Model(inputs=inputImage, outputs=x29)
m30 = Model(inputs=inputImage, outputs=x30)
m31 = Model(inputs=inputImage, outputs=x31)
m32 = Model(inputs=inputImage, outputs=x32)
m33 = Model(inputs=inputImage, outputs=x33)
m34 = Model(inputs=inputImage, outputs=x34)
m35 = Model(inputs=inputImage, outputs=x35)
m36 = Model(inputs=inputImage, outputs=x36)

model.summary()
encoder.summary()
m1.summary()
m2.summary()
##########################################

##########################################
LEARNING_RATE = 0.0005
n_epochs = 100
batch_size = parts
opt = Adam(lr = LEARNING_RATE)

def sse(y_true, y_pred):
    return K.sum(K.square(y_true - y_pred), axis = [1,2,3])

model.compile(optimizer=opt, loss = sse)

version = "v1_1_kaggle_modifiedloss"
if not os.path.exists(version):
  os.makedirs("./"+version)

checkpoint_model = ModelCheckpoint(os.path.join(version, '/weights.h5'), save_weights_only = True, verbose=1)

history = model.fit(data_flow, epochs=n_epochs, initial_epoch = 0, batch_size=batch_size,# steps_per_epoch=251, 
                    shuffle=True,
                    validation_data=data_flow,
                    callbacks = [
                        checkpoint_model,
                        EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=0.0001),
                        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),
                        TerminateOnNaN()])

model.save(version+'/model_'+version)
encoder.save(version+'/encoder_'+version)
m1.save(version+'/m1_'+version)
m2.save(version+'/m2_'+version)
m3.save(version+'/m3_'+version)
m4.save(version+'/m4_'+version)
m5.save(version+'/m5_'+version)
m6.save(version+'/m6_'+version)
m7.save(version+'/m7_'+version)
m8.save(version+'/m8_'+version)
m9.save(version+'/m9_'+version)
m10.save(version+'/m10_'+version)
m11.save(version+'/m11_'+version)
m12.save(version+'/m12_'+version)
m13.save(version+'/m13_'+version)
m14.save(version+'/m14_'+version)
m15.save(version+'/m15_'+version)
m16.save(version+'/m16_'+version)
m17.save(version+'/m17_'+version)
m18.save(version+'/m18_'+version)
m19.save(version+'/m19_'+version)
m20.save(version+'/m20_'+version)
m21.save(version+'/m21_'+version)
m22.save(version+'/m22_'+version)
m23.save(version+'/m23_'+version)
m24.save(version+'/m24_'+version)
m25.save(version+'/m25_'+version)
m26.save(version+'/m26_'+version)
m27.save(version+'/m27_'+version)
m28.save(version+'/m28_'+version)
m29.save(version+'/m29_'+version)
m30.save(version+'/m30_'+version)
m31.save(version+'/m31_'+version)
m32.save(version+'/m32_'+version)
m33.save(version+'/m33_'+version)
m34.save(version+'/m34_'+version)
m35.save(version+'/m35_'+version)
m36.save(version+'/m36_'+version)

np.save(version+'/history_'+version+'.npy', model.history.history)
##########################################
