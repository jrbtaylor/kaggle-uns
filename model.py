# -*- coding: utf-8 -*-
"""
Created on Tue May 31 23:45:55 2016

resnet modified from: https://github.com/raghakot/keras-resnet/blob/master/resnet.py
resnet as proposed in: http://arxiv.org/pdf/1603.05027v1.pdf
fire modules/squeezenet as in: http://arxiv.org/pdf/1602.07360v3.pdf
unet architecture as in: https://arxiv.org/pdf/1505.04597.pdf

@author: jason
"""

from keras.models import Model
from keras.layers import Input, merge, Dropout, Activation, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Reshape
from keras.layers.normalization import BatchNormalization
from data import rows, cols


def _shortcut(input,residual):
    equal_features = residual._keras_shape[1]==input._keras_shape[1]
    shortcut = input
    if not equal_features:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1],nb_row=1,nb_col=1)(input)
    return merge([shortcut,residual],mode='sum')


def _bn_relu_conv(nb_filter,filt_size):
    def f(input):
        norm = BatchNormalization()(input)
        relu = Activation('relu')(norm)
        return Convolution2D(nb_filter=nb_filter,nb_row=filt_size,nb_col=filt_size,border_mode='same')(relu)
    return f
    
    
def _conv_relu_bn(nb_filter,filt_size):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter,nb_row=filt_size,nb_col=filt_size,border_mode='same')(input)
        relu = Activation('relu')(conv)
        return BatchNormalization()(relu)
    return f
    

def _res_block(nb_filter):
    def f(input):
        residual = _conv_relu_bn(nb_filter,3)(input)
        residual = _conv_relu_bn(nb_filter,3)(residual)
        return _shortcut(input,residual)
    return f


def _dice_loss(y_true, y_pred):
    from keras import backend as K
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return -(2. * K.dot(y_true_f, K.transpose(y_pred_f)) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    
def _dice(y_true,y_pred):
    y_pred = (y_pred>0.5).astype('float32')
    return -_dice_loss(y_true,y_pred)
        
        
def init_res(f):
    inputs = Input((1,rows,cols))
    conv1 = Convolution2D(f,3,3,activation='relu',border_mode='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(f,3,3,activation='relu',border_mode='same')(conv1)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv2)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv3)
    conv4 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv4)
    conv5 = MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(16*f,3,3,activation='relu',border_mode='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(16*f,3,3,activation='relu',border_mode='same')(conv5)
#    dense = Flatten()(conv5)
#    dense = Dense(64,activation='relu')(dense)
#    dense = Dense(1,activation='hard_sigmoid')(dense)
    
    up1 = merge([UpSampling2D(size=(2,2))(conv5),conv4],mode='concat',concat_axis=1)
    conv6 = BatchNormalization()(up1)
    conv6 = _res_block(8*f)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = _res_block(8*f)(conv6)
    conv6 = _res_block(8*f)(conv6)
    up2 = merge([UpSampling2D(size=(2,2))(conv6),conv3],mode='concat',concat_axis=1)
    conv7 = BatchNormalization()(up2)
    conv7 = _res_block(4*f)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = _res_block(4*f)(conv7)
    conv7 = _res_block(4*f)(conv7)
    up3 = merge([UpSampling2D(size=(2,2))(conv7),conv2],mode='concat',concat_axis=1)
    conv8 = BatchNormalization()(up3)
    conv8 = _res_block(2*f)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = _res_block(2*f)(conv8)
    conv8 = _res_block(2*f)(conv8)
    up4 = merge([UpSampling2D(size=(2,2))(conv8),conv1],mode='concat',concat_axis=1)
    conv9 = BatchNormalization()(up4)
    conv9 = _res_block(f)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = _res_block(f)(conv9)
    
    outputs = Convolution2D(1,3,3,activation='hard_sigmoid',border_mode='same')(conv9)
#    outputs = Convolution2D(1,11,11,activation='hard_sigmoid',border_mode='same')(outputs)
    
    net = Model(input=inputs,output=outputs)

    net.compile(loss=_dice_loss,optimizer='adadelta',metrics=[_dice])
    
    return net

def init_fractalunet():
    f = 16    
    
    inputs = Input((1,rows,cols))
    
    conv1 = Convolution2D(f,3,3,activation='relu',border_mode='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(f,3,3,activation='relu',border_mode='same')(conv1)
    
    down1 = MaxPooling2D(pool_size=(2,2))(conv1)
    
    conv2 = BatchNormalization()(down1)
    conv2 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv2)
    
    down2 = MaxPooling2D(pool_size=(2,2))(conv2)
    
    conv3 = BatchNormalization()(down2)
    conv3 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv3)
    
    down3 = MaxPooling2D(pool_size=(2,2))(conv3)
    
    conv4 = BatchNormalization()(down3)
    conv4 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv4)
    
    down4 = MaxPooling2D(pool_size=(2,2))(conv4)
    
    conv5 = BatchNormalization()(down4)
    conv5 = Convolution2D(16*f,3,3,activation='relu',border_mode='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(16*f,3,3,activation='relu',border_mode='same')(conv5)
    
    up1 = merge([UpSampling2D(size=(2,2))(conv5),conv4],mode='concat',concat_axis=1)
    
    conv6 = BatchNormalization()(up1)
    conv6 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv6)
    
    up2 = merge([UpSampling2D(size=(2,2))(conv6),conv3],mode='concat',concat_axis=1)
    
    conv7 = BatchNormalization()(up2)
    conv7 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv7)
    
    up3 = merge([UpSampling2D(size=(2,2))(conv7),conv2],mode='concat',concat_axis=1)
    
    conv8 = BatchNormalization()(up3)
    conv8 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv8)
    
    up4 = merge([UpSampling2D(size=(2,2))(conv8),conv1],mode='concat',concat_axis=1)
    
    conv9 = BatchNormalization()(up4)
    conv9 = Convolution2D(f,3,3,activation='relu',border_mode='same')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Convolution2D(f,3,3,activation='relu',border_mode='same')(conv9)
    
    # --- end first u block
    
    down1b = MaxPooling2D(pool_size=(2,2))(conv9)
    down1b = merge([down1b,conv8],mode='concat',concat_axis=1)
    
    conv2b = BatchNormalization()(down1b)
    conv2b = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv2b)
    conv2b = BatchNormalization()(conv2b)
    conv2b = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv2b)
    
    down2b = MaxPooling2D(pool_size=(2,2))(conv2b)
    down2b = merge([down2b,conv7],mode='concat',concat_axis=1)
    
    conv3b = BatchNormalization()(down2b)
    conv3b = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv3b)
    conv3b = BatchNormalization()(conv3b)
    conv3b = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv3b)
    
    down3b = MaxPooling2D(pool_size=(2,2))(conv3b)
    down3b = merge([down3b,conv6],mode='concat',concat_axis=1)
    
    conv4b = BatchNormalization()(down3b)
    conv4b = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv4b)
    conv4b = BatchNormalization()(conv4b)
    conv4b = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv4b)
    
    down4b = MaxPooling2D(pool_size=(2,2))(conv4b)
    down4b = merge([down4b,conv5],mode='concat',concat_axis=1)
    
    conv5b = BatchNormalization()(down4b)
    conv5b = Convolution2D(16*f,3,3,activation='relu',border_mode='same')(conv5b)
    conv5b = BatchNormalization()(conv5b)
    conv5b = Convolution2D(16*f,3,3,activation='relu',border_mode='same')(conv5b)
    
    up1b = merge([UpSampling2D(size=(2,2))(conv5b),conv4b],mode='concat',concat_axis=1)
    
    conv6b = BatchNormalization()(up1b)
    conv6b = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv6b)
    conv6b = BatchNormalization()(conv6b)
    conv6b = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv6b)
    
    up2b = merge([UpSampling2D(size=(2,2))(conv6b),conv3b],mode='concat',concat_axis=1)
    
    conv7b = BatchNormalization()(up2b)
    conv7b = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv7b)
    conv7b = BatchNormalization()(conv7b)
    conv7b = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv7b)
    
    up3b = merge([UpSampling2D(size=(2,2))(conv7b),conv2b],mode='concat',concat_axis=1)
    
    conv8b = BatchNormalization()(up3b)
    conv8b = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv8b)
    conv8b = BatchNormalization()(conv8b)
    conv8b = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv8b)
    
    up4b = merge([UpSampling2D(size=(2,2))(conv8b),conv9],mode='concat',concat_axis=1)
    
    conv9b = BatchNormalization()(up4b)
    conv9b = Convolution2D(f,3,3,activation='relu',border_mode='same')(conv9b)
    conv9b = BatchNormalization()(conv9b)
    conv9b = Convolution2D(f,3,3,activation='relu',border_mode='same')(conv9b)
    conv9b = BatchNormalization()(conv9b)
    
    outputs = Convolution2D(1,1,1,activation='hard_sigmoid',border_mode='same')(conv9b)
#    outputs = Convolution2D(1,11,11,activation='hard_sigmoid',border_mode='same')(outputs)
    
    net = Model(input=inputs,output=outputs)

    net.compile(loss=_dice_loss,optimizer='adadelta',metrics=[_dice])
    
    return net
        
    
def init_conv():    
    inputs = Input((1,rows,cols))
    conv1 = Convolution2D(32,3,3,activation='relu',border_mode='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(32,3,3,activation='relu',border_mode='same')(conv1)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv2)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(128,3,3,activation='relu',border_mode='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(128,3,3,activation='relu',border_mode='same')(conv3)
    conv4 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv4)
    conv5 = MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv5)
#    dense = Flatten()(conv5)
#    dense = Dense(64,activation='relu')(dense)
#    dense = Dense(1,activation='hard_sigmoid')(dense)
    
    up1 = merge([UpSampling2D(size=(2,2))(conv5),conv4],mode='concat',concat_axis=1)
    conv6 = BatchNormalization()(up1)
    conv6 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv6)
    up2 = merge([UpSampling2D(size=(2,2))(conv6),conv3],mode='concat',concat_axis=1)
    conv7 = BatchNormalization()(up2)
    conv7 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv7)
    up3 = merge([UpSampling2D(size=(2,2))(conv7),conv2],mode='concat',concat_axis=1)
    conv8 = BatchNormalization()(up3)
    conv8 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv8)
    up4 = merge([UpSampling2D(size=(2,2))(conv8),conv1],mode='concat',concat_axis=1)
    conv9 = Dropout(0.125)(up4)
    conv9 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv9)
    conv9 = Dropout(0.25)(conv9)
    outputs = Convolution2D(1,1,1,activation='hard_sigmoid',border_mode='same')(conv9)
    outputs = Convolution2D(1,11,11,activation='hard_sigmoid',border_mode='same')(outputs)
    
    net = Model(input=inputs,output=outputs)

    net.compile(loss=_dice_loss,optimizer='adadelta',metrics=[_dice])
    
    return net
    

def init_conv_old(f): 
    inputs = Input((1,rows,cols))
    conv1 = Convolution2D(f,3,3,activation='relu',border_mode='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(f,3,3,activation='relu',border_mode='same')(conv1)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv2)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv3)
    conv4 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv4)
    conv5 = MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(16*f,3,3,activation='relu',border_mode='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(16*f,3,3,activation='relu',border_mode='same')(conv5)
#    dense = Flatten()(conv5)
#    dense = Dense(64,activation='relu')(dense)
#    dense = Dense(1,activation='hard_sigmoid')(dense)
    
    up1 = merge([UpSampling2D(size=(2,2))(conv5),conv4],mode='concat',concat_axis=1)
    conv6 = BatchNormalization()(up1)
#    conv6 = Dropout(0.3)(up1)
    conv6 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv6)
#    conv6 = Dropout(0.3)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Convolution2D(8*f,3,3,activation='relu',border_mode='same')(conv6)
    up2 = merge([UpSampling2D(size=(2,2))(conv6),conv3],mode='concat',concat_axis=1)
#    conv7 = Dropout(0.3)(up2)
    conv7 = BatchNormalization()(up2)
    conv7 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv7)
#    conv7 = Dropout(0.3)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(4*f,3,3,activation='relu',border_mode='same')(conv7)
    up3 = merge([UpSampling2D(size=(2,2))(conv7),conv2],mode='concat',concat_axis=1)
#    conv8 = Dropout(0.3)(up3)
    conv8 = BatchNormalization()(up3)
    conv8 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv8)
#    conv8 = Dropout(0.2)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Convolution2D(2*f,3,3,activation='relu',border_mode='same')(conv8)
    up4 = merge([UpSampling2D(size=(2,2))(conv8),conv1],mode='concat',concat_axis=1)
#    conv9 = Dropout(0.3)(up4)
    conv9 = BatchNormalization()(up4)
    conv9 = Convolution2D(f,3,3,activation='relu',border_mode='same')(conv9)
#    conv9 = Dropout(0.4)(conv9)
    conv9 = BatchNormalization()(conv9)
    outputs = Convolution2D(1,1,1,activation='hard_sigmoid',border_mode='same')(conv9)
    outputs = Convolution2D(1,11,11,activation='hard_sigmoid',border_mode='same')(outputs)
    
    net = Model(input=inputs,output=outputs)

    net.compile(loss=_dice_loss,optimizer='adadelta',metrics=[_dice])
    
    return net