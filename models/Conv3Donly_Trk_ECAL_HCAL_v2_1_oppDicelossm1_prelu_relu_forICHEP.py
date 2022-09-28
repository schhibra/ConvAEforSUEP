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
def get_file_list(path):
    flist = []
    flist += glob.glob(path + '/' + '*.h5')
    flist.sort()
    print("flist: ", flist)
    return flist

def read_images_from_file(fname):
    print("\nAppending %s" %fname)
    with h5py.File(fname,'r') as f:
        ImageTrk  = np.array(f.get("ImageTrk_PUcorr")[:], dtype=np.float32)
        ImageTrk  = ImageTrk.reshape(ImageTrk.shape[0], ImageTrk.shape[1], ImageTrk.shape[2], 1)

        ImageECAL = np.array(f.get("ImageECAL")[:], dtype=np.float32)
        ImageECAL = ImageECAL.reshape(ImageECAL.shape[0], ImageECAL.shape[1], ImageECAL.shape[2], 1)

        ImageHCAL = np.array(f.get("ImageHCAL")[:], dtype=np.float32)
        ImageHCAL = ImageHCAL.reshape(ImageHCAL.shape[0], ImageHCAL.shape[1], ImageHCAL.shape[2], 1)

        Image3D = np.concatenate([ImageTrk, ImageECAL, ImageHCAL], axis=-1)

        Image3D_zero = np.zeros((Image3D.shape[0], 288, 360, 3), dtype=np.float32)
        Image3D_zero[:, 1:287, :, :] += Image3D
        Image3D_zero = np.divide(Image3D_zero, 2000., dtype=np.float32)
        return Image3D_zero
        
def concatenate_by_file_content(Image3D, fname):
    Image3D_tmp = read_images_from_file(fname)
    Image3D = np.concatenate([Image3D, Image3D_tmp], axis=0) if Image3D.size else Image3D_tmp
    return Image3D
    
def gen(parts_n, pathindex):
    
    if   (pathindex == 0): flist = get_file_list("/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_qcd_v1/gensim/output/train")
    elif (pathindex == 1): flist = get_file_list("/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_qcd_v1/gensim/output/val"  )

    Image3D_conc = np.array([])
    
    for i_file, fname in enumerate(flist):
        Image3D_conc = concatenate_by_file_content(Image3D_conc, fname)
        
        while (len(Image3D_conc) >= parts_n):
            Image3D_part, Image3D_conc = Image3D_conc[:parts_n] , Image3D_conc[parts_n:]
            #print (" np.sum ", np.sum(Image3D_part[0]))
            #print (" ",Image3D_part.shape, Image3D_conc.shape)
            Image3D_part_tf = tf.convert_to_tensor(Image3D_part, dtype=tf.float32)
            yield (Image3D_part_tf, Image3D_part_tf)

parts = 128
dataset_train = tf.data.Dataset.from_generator(
    gen, 
    args=[parts, 0],
    output_types=(tf.float32, tf.float32))

dataset_val = tf.data.Dataset.from_generator(
    gen, 
    args=[parts, 1],
    output_types=(tf.float32, tf.float32))
##########################################

##########################################
# keras imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, PReLU, BatchNormalization, Activation
#from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Lambda, Layer, LeakyReLU, AveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
#from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
##########################################

##########################################
tf.keras.backend.set_floatx('float32')

img_rows = 288
img_cols = 360
img_chns = 3

############ENCODER
inputImage = Input(shape=(img_rows, img_cols, img_chns))
x1 = Conv2D(128, (3,3), strides=(3, 3), padding="same")(inputImage)
x2 = BatchNormalization()(x1)
x3 = PReLU()(x2)
x4 = Conv2D(64, (3,3), strides=(2, 2), padding="same")(x3)
x5 = BatchNormalization()(x4)
x6 = PReLU()(x5)
x7 = Conv2D(32, (3,3), strides=(2, 2), padding="same")(x6)
x8 = BatchNormalization()(x7)
x9 = PReLU()(x8)
x10 = Conv2D(16, (3,3), strides=(2, 2), padding="same")(x9)
x11 = BatchNormalization()(x10)
x12 = PReLU()(x11)
x13 = Conv2D(8, (3,3), strides=(2, 3), padding="same")(x12)
x14 = BatchNormalization()(x13)
encoder_output = PReLU()(x14)

############DECODER                                                                                                                                                                                   
x15 = Conv2DTranspose(16, (3,3), strides=(2, 3), padding="same")(encoder_output)
x16 = BatchNormalization()(x15)
x17 = PReLU()(x16)
x18 = Conv2DTranspose(32, (3,3), strides=(2, 2), padding="same")(x17)
x19 = BatchNormalization()(x18)
x20 = PReLU()(x19)
x21 = Conv2DTranspose(64, (3,3), strides=(2, 2), padding="same")(x20)
x22 = BatchNormalization()(x21)
x23 = PReLU()(x22)
x24 = Conv2DTranspose(128, (3,3), strides=(2, 2), padding="same")(x23)
x25 = BatchNormalization()(x24)
x26 = PReLU()(x25)
x27 = Conv2DTranspose(img_chns, (3,3), strides=(3, 3), padding="same")(x26)
x28 = BatchNormalization()(x27)
output = Activation('relu')(x28)

model = Model(inputs=inputImage, outputs=output)
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

model.summary()
encoder.summary()
m1.summary()
m2.summary()
##########################################

##########################################
LEARNING_RATE = 0.001 #default 0.001
n_epochs = 100
batch_size = parts
opt = Adam(lr = LEARNING_RATE)

tf.config.run_functions_eagerly(True)

def Diceloss(y_true, y_pred, smooth=1e-6):

    dice = []
    for i in range(parts):
        y_true_tmp = tf.reshape(y_true[i], shape=(1, (288*360*3)))
        y_pred_tmp = tf.reshape(y_pred[i], shape=(1, (288*360*3)))

        idx_keep_in = tf.where(y_true_tmp[0,:]>0)[:,-1]
        y_true_tmp  = tf.gather(y_true_tmp[0,:], idx_keep_in)
        y_pred_tmp  = tf.gather(y_pred_tmp[0,:], idx_keep_in)

        y_true_tmp = tf.reshape(y_true_tmp, shape=(1, y_true_tmp.shape[0]))
        y_pred_tmp = tf.reshape(y_pred_tmp, shape=(y_pred_tmp.shape[0], 1))

        intersection = K.sum(K.dot(y_true_tmp, y_pred_tmp))
        dice.append((K.sum(K.square(y_true[i])) + K.sum(K.square(y_pred[i])) + smooth) / (2 * intersection + smooth) - 1)
    return dice

def intersection(y_true, y_pred):

    intersection = []
    for i in range(parts):
        y_true_tmp = tf.reshape(y_true[i], shape=(1, (288*360*3)))
        y_pred_tmp = tf.reshape(y_pred[i], shape=(1, (288*360*3)))

        idx_keep_in = tf.where(y_true_tmp[0,:]>0)[:,-1]
        y_true_tmp  = tf.gather(y_true_tmp[0,:], idx_keep_in)
        y_pred_tmp  = tf.gather(y_pred_tmp[0,:], idx_keep_in)

        y_true_tmp = tf.reshape(y_true_tmp, shape=(1, y_true_tmp.shape[0]))
        y_pred_tmp = tf.reshape(y_pred_tmp, shape=(y_pred_tmp.shape[0], 1))

        intersection.append(K.sum(K.dot(y_true_tmp, y_pred_tmp)))
    return intersection

def sum_y_true(y_true, y_pred):
    return K.sum(y_true, axis=[1,2,3])

def sum_y_pred(y_true, y_pred):
    return K.sum(y_pred, axis=[1,2,3])

#model.compile(optimizer=opt, loss = Diceloss), metrics=[intersection, sum_y_true, sum_y_pred, 'accuracy'])
model.compile(optimizer=opt, loss = DiceLoss)

version = "v2_1_oppDiceloss_prelu_relu_repeat_100"
if not os.path.exists(version):
    os.makedirs("./"+version)
    
#checkpoint_model = ModelCheckpoint(os.path.join(version, '/weights.h5'), save_weights_only = True, verbose=1)

history = model.fit(dataset_train, epochs=n_epochs, initial_epoch = 0, batch_size=batch_size,
                    validation_data=dataset_val,
                    callbacks = [
#                        checkpoint_model,
                        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
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

np.save(version+'/history_'+version+'.npy', model.history.history)
##########################################
