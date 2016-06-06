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

#%%
# Model
import model
cnn = model.init_conv()

# Augmentation
#import augment

# Train
batch_size = 32
nb_epoch = 100
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint('cnn.hdf5', monitor='val_loss', save_best_only=True)
earlyStopping= EarlyStopping(monitor='val_loss', patience=5, verbose=0)
history = cnn.fit(x_train,y_train,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,callbacks=[checkpoint,earlyStopping],validation_split=0.04,shuffle=True)

#%%
# Test
cnn.load_weights('cnn.hdf5')
batch_size = 128
y_pred = cnn.predict(x_test,batch_size=batch_size,verbose=1)
import submit
submit.final(y_pred,idx_test)
