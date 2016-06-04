# -*- coding: utf-8 -*-
"""
Created on Tue May 31 23:45:55 2016

resnet modified from: https://github.com/raghakot/keras-resnet/blob/master/resnet.py

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

def _res_block(nb_filter):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filter,1)(input)
        conv_3_3 = _bn_relu_conv(nb_filter,3)(conv_1_1)
        residual = _bn_relu_conv(nb_filter*2,1)(conv_3_3)
        return _shortcut(input,residual)
    return f
    
def debug():
    inputs = Input((1,rows,cols))
    res1 = _res_block(32)(inputs)
    outputs = Convolution2D(1,1,1,activation='hard_sigmoid',border_mode='same')(res1)
    net = Model(input=inputs,output=outputs)
    return net
   
def init():
    inputs = Input((1,rows,cols))
    conv1 = Convolution2D(32,3,3,border_mode='same')(inputs)
    res1 = _res_block(16)(conv1)
    dwn1 = MaxPooling2D(pool_size=(2,2))(res1)
    res2 = _res_block(32)(dwn1)
    dwn2 = MaxPooling2D(pool_size=(2,2))(res2)
    res3 = _res_block(64)(dwn2)
    dwn3 = MaxPooling2D(pool_size=(2,2))(res3)
    res4 = _res_block(128)(dwn3)
    dwn4 = MaxPooling2D(pool_size=(2,2))(res4)
    res5 = _res_block(256)(dwn4)
    up5 = merge([UpSampling2D(size=(2,2))(res5),res4],mode='concat',concat_axis=1)
    res6 = _res_block(128)(up5)
    up6 = merge([UpSampling2D(size=(2,2))(res6),res3],mode='concat',concat_axis=1)
    res7 = _res_block(64)(up6)
    up7 = merge([UpSampling2D(size=(2,2))(res7),res2],mode='concat',concat_axis=1)
    res8 = _res_block(32)(up7)
    up8 = merge([UpSampling2D(size=(2,2))(res8),res1],mode='concat',concat_axis=1)
    res9 = _res_block(16)(up8)
    
    outputs = Convolution2D(1,1,1,activation='hard_sigmoid',border_mode='same')(res9)
    
    net = Model(input=inputs,output=outputs)
    
    # Loss function
    def dice_loss(y_true, y_pred):
        from keras import backend as K
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        return -(2. * K.dot(y_true_f, K.transpose(y_pred_f)) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        
    def dice(y_true,y_pred):
        y_pred = (y_pred>0.5).astype('float32')
        return -dice_loss(y_true,y_pred)

    net.compile(loss=dice_loss,optimizer='adadelta',metrics=[dice])
    
    return net


def init_old():
    
    
    inputs = Input((1,rows,cols))
    conv1 = Convolution2D(32,3,3,activation='relu',border_mode='same')(inputs)
#    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(32,3,3,activation='relu',border_mode='same')(conv1)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv1)
#    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv2)
#    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv2)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv2)
#    conv3 = BatchNormalization()(conv3)
#    conv3 = Dropout(0.3)(conv3)
    conv3 = Convolution2D(128,3,3,activation='relu',border_mode='same')(conv3)
#    conv3 = BatchNormalization()(conv3)
#    conv3 = Dropout(0.3)(conv3)
    conv3 = Convolution2D(128,3,3,activation='relu',border_mode='same')(conv3)
    conv4 = MaxPooling2D(pool_size=(2,2))(conv3)
#    conv4 = BatchNormalization()(conv4)
#    conv4 = Dropout(0.3)(conv4)
    conv4 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv4)
#    conv4 = BatchNormalization()(conv4)
#    conv4 = Dropout(0.3)(conv4)
    conv4 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv4)
    conv5 = MaxPooling2D(pool_size=(2,2))(conv4)
#    conv5 = BatchNormalization()(conv5)
#    conv5 = Dropout(0.3)(conv5)
    conv5 = Convolution2D(512,3,3,activation='relu',border_mode='same')(conv5)
#    conv5 = BatchNormalization()(conv5)
#    conv5 = Dropout(0.3)(conv5)
    conv5 = Convolution2D(512,3,3,activation='relu',border_mode='same')(conv5)
#    dense = Flatten()(conv5)
#    dense = Dense(64,activation='relu')(dense)
#    dense = Dense(1,activation='hard_sigmoid')(dense)
    up1 = Merge([UpSampling2D(size=(2,2))(conv5),conv4],mode='concat',concat_axis=1)
#    conv6 = Dropout(0.3)(up1)
    conv6 = Convolution2D(256,3,3,activation='relu',border_mode='same')(up1)
#    conv6 = Dropout(0.3)(conv6)
    conv6 = Convolution2D(256,3,3,activation='relu',border_mode='same')(conv6)
    up2 = Merge([UpSampling2D(size=(2,2))(conv6),conv3],mode='concat',concat_axis=1)
#    conv7 = Dropout(0.3)(up2)
    conv7 = Convolution2D(128,3,3,activation='relu',border_mode='same')(up2)
#    conv7 = Dropout(0.3)(conv7)
    conv7 = Convolution2D(128,3,3,activation='relu',border_mode='same')(conv7)
    up3 = Merge([UpSampling2D(size=(2,2))(conv7),conv2],mode='concat',concat_axis=1)
#    conv8 = Dropout(0.3)(up3)
    conv8 = Convolution2D(64,3,3,activation='relu',border_mode='same')(up3)
#    conv8 = Dropout(0.3)(conv8)
    conv8 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv8)
    up4 = Merge([UpSampling2D(size=(2,2))(conv8),conv1],mode='concat',concat_axis=1)
#    conv9 = Dropout(0.2)(up4)
    conv9 = Convolution2D(32,3,3,activation='relu',border_mode='same')(up4)
#    conv9 = Dropout(0.2)(conv9)
    outputs = Convolution2D(1,1,1,activation='hard_sigmoid',border_mode='same')(conv9)
    
    net = Model(input=inputs,output=outputs)
    
    # Loss function
    def dice_loss(y_true, y_pred):
        from keras import backend as K
        smooth = 1.
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        return -(2. * K.dot(y_true_f, K.transpose(y_pred_f)) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        
    def dice(y_true,y_pred):
        y_pred = (y_pred>0.5).astype('float32')
        return -dice_loss(y_true,y_pred)

    net.compile(loss=dice_loss,optimizer='adadelta',metrics=[dice])
    
    return net