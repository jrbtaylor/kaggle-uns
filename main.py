# -*- coding: utf-8 -*-
"""
Main function for Kaggle UNS Challenge

@author: jason
"""

import numpy as np
np.random.seed(1)  # for reproducibility
import os

# Load data
import data
if os.path.isfile('x_train.npy') and os.path.isfile('y_train.npy') and os.path.isfile('x_test.npy') and os.path.isfile('idx_test.npy'):
    print('Reloading data')
    def reloading(x):
        return np.load(os.path.join(os.getcwd(),x))
    x_train = reloading('x_train.npy')
    y_train = reloading('y_train.npy')
    x_test = reloading('x_test.npy')
    idx_test = reloading('idx_test.npy')
else:
    x_train,y_train = data.load_train()
    x_test,idx_test = data.load_test()
    x_train,x_test = data.normalize(x_train,x_test)
    
    # Save data for quick reloading
    np.save('x_train',x_train)
    np.save('y_train',y_train)
    np.save('x_test',x_test)
    np.save('idx_test',idx_test)
    
# Split train for validation
nb_val = int(np.round(0.04*x_train.shape[0]))
x_val = x_train[0:nb_val,:,:,:]
y_val = y_train[0:nb_val,:,:,:]
x_train = x_train[nb_val:,:,:,:]
y_train = y_train[nb_val:,:,:,:]


#%%
# Model
import model
reload(model)
cnn = model.init_conv()

# Train (no augmentation)
batch_size = 32
nb_epoch = 100
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint('cnn_fire.hdf5', monitor='val_loss', save_best_only=True)
earlyStopping= EarlyStopping(monitor='val_loss', patience=5, verbose=0)
history = cnn.fit(x_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,callbacks=[checkpoint,earlyStopping],validation_data=(x_val,y_val),shuffle=True)


#%% Model
import model
reload(model)
cnn = model.init_conv()

# Training w/ augmentation
batch_size = 32
nb_epoch = 100

import augment
reload(augment)
datagen = augment.Generator(rotation_range=10)
trainflow = datagen.flow(x_train,y_train,batch_size=batch_size,seed=1)

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint('cnn_fire.hdf5', monitor='val_loss', save_best_only=True)
earlyStopping= EarlyStopping(monitor='val_loss', patience=5, verbose=0)
history = cnn.fit_generator(trainflow,samples_per_epoch=x_train.shape[0],nb_epoch=nb_epoch,verbose=1,callbacks=[checkpoint,earlyStopping],validation_data=(x_val,y_val),max_q_size=10)


#%% Test
cnn.load_weights('cnn.hdf5')
batch_size = 256
y_pred = cnn.predict(x_test,batch_size=batch_size,verbose=1)
import submit
submit.final(y_pred,idx_test)
