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
reload(data)
if os.path.isfile('x_train.npy') and os.path.isfile('y_train.npy') and os.path.isfile('idx_train.npy') and os.path.isfile('x_test.npy') and os.path.isfile('idx_test.npy'):
    print('Reloading data')
    def reloading(x):
        return np.load(os.path.join(os.getcwd(),x))
    x_train = reloading('x_train.npy')
    y_train = reloading('y_train.npy')
    idx_train = reloading('idx_train.npy')
    x_test = reloading('x_test.npy')
    idx_test = reloading('idx_test.npy')
else:
    x_train,y_train,idx_train = data.load_train()
    x_test,idx_test = data.load_test()
    x_train,x_test = data.normalize(x_train,x_test)
    
    # Save data for quick reloading
    np.save('x_train',x_train)
    np.save('y_train',y_train)
    np.save('idx_train',idx_train)
    np.save('x_test',x_test)
    np.save('idx_test',idx_test)


#%%
import model; reload(model)
import augment; reload(augment)
from keras.callbacks import ModelCheckpoint, EarlyStopping

# For fractalnet, need to increase recursion limit
import sys
sys.setrecursionlimit(10000)

epoch_size = 1024
batch_size = 32
nb_epoch = 1000

nb_val = int(np.round(0.06*len(set(idx_train)))) # 6% is 3, 5% is 2 patients
ensemble = int(np.floor(len(set(idx_train))/nb_val)) # number of models to train

# Model
cnn = model.init_fractal(32,3,2,0.15)
#cnn = model.init_fractalunet(32)
#cnn = model.init_resMulti(32)
#cnn = model.init_resMultiDrop(32)

# Note: keras lacks a mechanism to re-initialize the weights without recompiling the model,
#       since this may take >10 minutes for a fractalnet, just shuffle the original weights
w0 = cnn.get_weights()

for e in range(1): #range(ensemble):
    
    # Re-initialize the weights
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in w0]
    cnn.set_weights(weights)
    
    # Validation/training split
    split = [e*nb_val+i+1 for i in range(nb_val)] # patient indeces in validation set
    val_set = [i for i in range(len(idx_train)) if int(idx_train[i]) in split]
    train_set = [i for i in range(len(idx_train)) if int(idx_train[i]) not in split]
    x_val = x_train[val_set,:,:,:]
    y_val = y_train[val_set,:,:,:]

    # Augmentation
    datagen = augment.Generator(hflip=True,vflip=True,rotation=10,zoom=0.05,shear=5)
#    datagen = augment.Generator(hflip=True,vflip=True)
    trainflow = datagen.flow(x_train[train_set,:,:,:],y_train[train_set,:,:,:],batch_size=batch_size,seed=1)

    # Train
    filename = 'cnn'+str(e)+'.hdf5'
    checkpoint = ModelCheckpoint(filename, monitor='val_loss', save_best_only=True)
    earlyStopping= EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    history = cnn.fit_generator(trainflow,samples_per_epoch=epoch_size,nb_epoch=nb_epoch,verbose=1,callbacks=[checkpoint,earlyStopping],validation_data=(x_val,y_val),max_q_size=10)
    
    if np.min(history.history['val_loss'])>-0.5: # if it failed to converge to anything useful
        os.remove(filename)
        print("Deleted model that failed to converge")  


#%%
# Test
#import model; reload(model)
#cnn = model.init_fractal(32,3,2,0.5)

batch_size = 32
modelcount = 0
y_pred = np.zeros_like(x_test)

for e in range(1):
    
    # Load the net
    filename = 'cnn'+str(e)+'.hdf5'
    if os.path.isfile(filename):
        cnn.load_weights(filename)
        modelcount = modelcount+1
        
        # calculate the predictions for this net
        y_pred = y_pred + cnn.predict(x_test,batch_size=batch_size,verbose=1)
        
# Average the predictions
y_pred = y_pred/modelcount

import postprocessing; reload(postprocessing)
y_pred = postprocessing.final(y_pred,y_train)
import submit
submit.final(y_pred,idx_test)
