# -*- coding: utf-8 -*-
"""
Created on Tue May 31 22:20:56 2016

@author: jason
"""

import numpy as np
np.random.seed(1)  # for reproducibility
import keras

# Options
opt = {'rows':56, 'cols':80}

# Load data
import data
x_train,y_train = data.load_train(opt)
x_test = data.load_test(opt)

#%%
# Model
import model
cnn = model.init(opt)

# Train
batch_size = 64
nb_epoch = 100
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('cnn.hdf5', monitor='loss', save_best_only=True)
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
history = cnn.fit(x,y,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,callbacks=[earlyStopping],validation_split=0.04,shuffle=True)

# Test

