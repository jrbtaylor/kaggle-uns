# -*- coding: utf-8 -*-
"""
Main function for Kaggle UNS Challenge

@author: jason
"""

import numpy as np
np.random.seed(1)  # for reproducibility
import os
import data

import model
import augment
from keras.callbacks import ModelCheckpoint, EarlyStopping

x_train0,y_train0,idx_train0,x_test,idx_test = data.load_data()

# fix missing masks in duplicate images
import duplicates
y_train0 = duplicates.dupmask(x_train0,y_train0,idx_train0,15)

# For fractalnet, need to increase recursion limit
import sys
sys.setrecursionlimit(10000)

epoch_size = 1024
batch_size = 16
nb_epoch = 1000
split_by_patient = False

if split_by_patient:
    nb_val = int(np.round(0.06*len(set(idx_train0)))) # 6% is 3, 5% is 2 patients
    ensemble = int(np.floor(len(set(idx_train0))/nb_val)) # number of models to train
else:
    nb_val = int(np.round(0.05*x_train0.shape[0]))
    ensemble = int(np.floor(x_train0.shape[0]/nb_val))
    shuffle = np.random.permutation(x_train0.shape[0])

#------------------------------------------------------------------------------
# Segmentation model
#------------------------------------------------------------------------------

#cnn = model.init_fractal2(40,3,2,0.15)

# Note: keras lacks a mechanism to re-initialize the weights without recompiling the model,
#       since this may take >10 minutes for a fractalnet, just shuffle the original weights
#w0 = cnn.get_weights()
for e in range(ensemble): # range(ensemble) or range(1)
    
#    # Re-initialize the weights
#    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in w0]
#    cnn.set_weights(weights)
    cnn = model.init_fractal2(32,3,2,0.15)
    
    # Validation/training split
    if split_by_patient:
        split = [e*nb_val+i+1 for i in range(nb_val)] # patient indeces in validation set
        val_set = [i for i in range(len(idx_train0)) if int(idx_train0[i]) in split]
        train_set = [i for i in range(len(idx_train0)) if int(idx_train0[i]) not in split]
        x_val = x_train0[val_set,:,:,:]
        y_val = y_train0[val_set,:,:,:]
        x_train = x_train0[train_set,:,:,:]
        y_train = y_train0[train_set,:,:,:]
    else:
        val_set = [e*nb_val+i+1 for i in range(nb_val)] # indeces in shuffle set
        train_set = [i for i in range(x_train0.shape[0]) if i not in val_set]
        x_val = x_train0[shuffle[val_set],:,:,:]
        y_val = y_train0[shuffle[val_set],:,:,:]
        x_train = x_train0[shuffle[train_set],:,:,:]
        y_train = y_train0[shuffle[train_set],:,:,:]    

    # Augmentation
    datagen = augment.Generator(hflip=True,vflip=True,rotation=20,zoom=0.05,shear=5)
    trainflow = datagen.flow(x_train,y_train,batch_size=batch_size,seed=1)

    # Train
    filename = 'cnn'+str(e)+'.hdf5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', save_best_only=True)
    earlyStopping= EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    history = cnn.fit_generator(trainflow,samples_per_epoch=epoch_size,nb_epoch=nb_epoch,verbose=1,callbacks=[checkpoint,earlyStopping],validation_data=(x_val,y_val),max_q_size=10)
    
    if np.min(history.history['val_loss'])>-0.58: # if it failed to converge to anything useful
        os.remove(filename)
        print("Deleted model that failed to converge")  

#------------------------------------------------------------------------------
# Predict likelihood of human labeling
#------------------------------------------------------------------------------
y_train0 = np.any(y_train0,axis=(1,2,3))

epoch_size = 4096
batch_size = 16
nb_epoch = 5000
split_by_patient = False

if split_by_patient:
    nb_val = int(np.round(0.06*len(set(idx_train0)))) # 6% is 3, 5% is 2 patients
    ensemble = int(np.floor(len(set(idx_train0))/nb_val)) # number of models to train
else:
    nb_val = int(np.round(0.05*x_train0.shape[0]))
    ensemble = int(np.floor(x_train0.shape[0]/nb_val))
    shuffle = np.random.permutation(x_train0.shape[0])

for e in range(ensemble):
    cnn = model.prob_label(48)
    
    # Validation/training split
    if split_by_patient:
        split = [e*nb_val+i+1 for i in range(nb_val)] # patient indeces in validation set
        val_set = [i for i in range(len(idx_train0)) if int(idx_train0[i]) in split]
        train_set = [i for i in range(len(idx_train0)) if int(idx_train0[i]) not in split]
        x_val = x_train0[val_set,:,:,:]
        y_val = y_train0[val_set,:,:,:]
        x_train = x_train0[train_set,:,:,:]
        y_train = y_train0[train_set,:,:,:]
    else:
        val_set = [e*nb_val+i+1 for i in range(nb_val)] # indeces in shuffle set
        train_set = [i for i in range(x_train0.shape[0]) if i not in val_set]
        x_val = x_train0[shuffle[val_set],:,:,:]
        y_val = y_train0[shuffle[val_set],:,:,:]
        x_train = x_train0[shuffle[train_set],:,:,:]
        y_train = y_train0[shuffle[train_set],:,:,:]
    
    # Augmentation
    datagen = augment.Generator(hflip=True,vflip=True,rotation=10,zoom=0.05,shear=5)
    trainflow = datagen.flow(x_train,y_train,batch_size=batch_size,seed=1)

    # Train
    filename = 'cnn_prob'+str(e)+'.hdf5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', save_best_only=True)
    earlyStopping= EarlyStopping(monitor='val_loss', patience=50, verbose=1)
    history = cnn.fit_generator(trainflow,samples_per_epoch=epoch_size,nb_epoch=nb_epoch,verbose=1,callbacks=[checkpoint,earlyStopping],validation_data=(x_val,y_val),max_q_size=10)
    
    if np.min(history.history['val_loss'])>0.5: # if it failed to converge to anything useful
        os.remove(filename)
        print("Deleted model that failed to converge")  

#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------

batch_size = 32

# Predict the segmentation masks
cnn = model.init_fractal2(40,3,2,0.15)
modelcount = 0
y_pred = np.zeros_like(x_test)

for e in range(ensemble):
    filename = 'cnn'+str(e)+'.hdf5'
    if os.path.isfile(filename):
        cnn.load_weights(filename)
        modelcount = modelcount+1
        y_pred = y_pred + cnn.predict(x_test,batch_size=batch_size,verbose=1)
        
y_pred = y_pred/modelcount

merge_with_probability = True

if merge_with_probability:
    # Predict human labeling error
    cnn = model.prob_label(48)
    modelcount = 0
    prob_label = np.zeros(x_test.shape[0])
    for e in range(ensemble):
        filename = 'cnn_prob'+str(e)+'.hdf5'
        if os.path.isfile(filename):
            cnn.load_weights(filename)
            modelcount = modelcount+1
            prob_label = prob_label + np.squeeze(cnn.predict(x_test,batch_size=batch_size,verbose=1))       
    prob_label = prob_label/modelcount
    prob_label = (prob_label>0.5).astype(float)
    
    # Combine segmentation masks and label probability
    for e in range(3):
        prob_label = np.expand_dims(prob_label,axis=1)
    y_pred = np.sqrt(np.multiply(y_pred,prob_label))

import postprocessing
y_pred = postprocessing.final(y_pred,y_train)
import submit
submit.final(y_pred,idx_test)
