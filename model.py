# -*- coding: utf-8 -*-
"""
Created on Tue May 31 23:45:55 2016

@author: jason
"""

def init(opt):
    from keras.models import Sequential
    from keras.layers import Dropout, Activation
    from keras.layers import Convolution2D, Reshape
    
    net = Sequential()
    net.add(Convolution2D(16, 5, 5,
                        border_mode='same',
                        input_shape=(1, opt['rows'], opt['cols'])))
    net.add(Activation('relu'))
    net.add(Convolution2D(32,5,5,border_mode='same'))
    net.add(Activation('relu'))
    net.add(Convolution2D(64,3,3,border_mode='same'))
    net.add(Activation('relu'))
    net.add(Convolution2D(128,3,3,border_mode='same'))
    net.add(Activation('relu'))
    net.add(Convolution2D(2,3,3,border_mode='same'))
    net.add(Reshape((opt['rows']*opt['cols'],2)))
    net.add(Activation('softmax'))
    
    return net