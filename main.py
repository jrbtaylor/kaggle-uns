# -*- coding: utf-8 -*-
"""
Main function for Kaggle UNS Challenge

@author: jason
"""

import numpy as np
np.random.seed(1)  # for reproducibility
import keras
import os

# Load data
import data
if os.path.isfile('x_train.npy') and os.path.isfile('y_train.npy') and os.path.isfile('x_test') and os.path.isfile('idx_test'):
    print('Reloading data')
    x_train = np.load('x_train')
    y_train = np.load('y_train')
    x_test = np.load('x_test')
    idx_test = np.load('idx_test')
else:
    x_train,y_train = data.load_train()
    x_test,idx_test = data.load_test()
    x_train,x_test = data.normalize(x_train,x_test)
    
    # Save data for quick reloading
    np.save('x_train',x_train)
    np.save('y_train',y_train)
    np.save('x_test',x_test)
    np.save('idx_test',idx_test)

#%%
# Model
import model
cnn = model.init()

# Augmentation
#import augment

# Train
batch_size = 32
nb_epoch = 100
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('cnn.hdf5', monitor='val_loss', save_best_only=True)
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)
history = cnn.fit(x_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,callbacks=[checkpoint,earlyStopping],validation_split=0.04,shuffle=True)

#%%
# Test
cnn.load_weights('cnn.hdf5')
y_pred = cnn.predict(x_test,batch_size=batch_size,verbose=1)
import submit
submit.final(y_pred,idx_test)
