# -*- coding: utf-8 -*-
"""
Created on Tue May 31 23:45:55 2016

@author: jason
"""

def init(opt):
    from keras.models import Model
    from keras.layers import Input, merge, Dropout, Activation
    from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Reshape
    from keras.layers.normalization import BatchNormalization
    
    inputs = Input((1,opt['rows'],opt['cols']))
    conv1 = Convolution2D(16,3,3,activation='relu',border_mode='same')(inputs)
    conv1 = Convolution2D(16,3,3,activation='relu',border_mode='same')(conv1)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Convolution2D(32,3,3,activation='relu',border_mode='same')(conv2)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv3)
    up1 = merge([UpSampling2D(size=(2, 2))(conv3), conv2], mode='concat', concat_axis=1)
    conv4 = Convolution2D(64,3,3,activation='relu',border_mode='same')(up1)
    up2 = merge([UpSampling2D(size=(2, 2))(conv4), conv1], mode='concat', concat_axis=1)
    conv5 = Convolution2D(64,3,3,activation='relu',border_mode='same')(up2)
    conv5 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv5)
    outputs = Convolution2D(2,3,3,border_mode='same')(conv5)
    outputs = Reshape((opt['rows']*opt['cols'],2))(outputs)
    outputs = Activation('softmax')(outputs)
    
    net = Model(input=inputs,output=outputs)
    
    # Loss function
    def dice_loss(y_true, y_pred):
        from keras import backend as K
        eps = 1
        y_true_f = K.flatten(y_true[...,1])
        y_pred_f = K.flatten(y_pred[...,1])
        return -(2. * K.dot(y_true_f, K.transpose(y_pred_f)) + eps) / (K.sum(y_true_f) + K.sum(y_pred_f) + eps)

    net.compile(loss='categorical_crossentropy',optimizer='adadelta',
              metrics=['accuracy',dice_loss])
    
    return net