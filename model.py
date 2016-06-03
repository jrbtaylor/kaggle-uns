# -*- coding: utf-8 -*-
"""
Created on Tue May 31 23:45:55 2016

@author: jason
"""

def init():
    from keras.models import Model
    from keras.layers import Input, merge, Dropout, Activation
    from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, Reshape
    from keras.layers.normalization import BatchNormalization
    from keras.layers.noise import GaussianDropout
    from data import rows, cols
    
    inputs = Input((1,rows,cols))
#    conv1 = GaussianDropout(0.1)(inputs)
    conv1 = Convolution2D(16,3,3,activation='relu',border_mode='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Convolution2D(16,3,3,activation='relu',border_mode='same')(conv1)
    conv2 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(32,3,3,activation='relu',border_mode='same')(conv2)
    conv3 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv3)
    conv4 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv4)        
    up1 = merge([UpSampling2D(size=(2,2))(conv4),conv3],mode='concat',concat_axis=1)
    conv5 = BatchNormalization()(up1)
    conv5 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv5)
    up2 = merge([UpSampling2D(size=(2,2))(conv5),conv2],mode='concat',concat_axis=1)
    conv6 = BatchNormalization()(up2)
    conv6 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv6)
    up3 = merge([UpSampling2D(size=(2,2))(conv6),conv1],mode='concat',concat_axis=1)
    conv7 = BatchNormalization()(up3)
    conv7 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv7)
    conv7 = Dropout(0.5)(conv7)
    conv7 = Convolution2D(64,3,3,activation='relu',border_mode='same')(conv7)
    conv7 = Dropout(0.5)(conv7)
    outputs = Convolution2D(1,3,3,activation='hard_sigmoid',border_mode='same')(conv7)
    
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