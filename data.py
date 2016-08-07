# -*- coding: utf-8 -*-
"""
Loads Kaggle Ultrasound Nerve Segmentation data

@author: jason
"""

import os
import numpy as np
from keras.preprocessing import image
from scipy.misc import imresize

rows = 96 #48 #96 #64
cols = 128 #64 #128 #80

def normalize(x_train,x_test):
    print('Normalizing data')
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train-mean)/std
    x_test = (x_test-mean)/std
    return x_train,x_test
    

# align x and y files so the image and labels correspond
def _align_files(x,y):
    def _rem_mask(y):
        return y.split('_mask')[0]    
    def _rem_tif(x):
        return x.split('.')[0]
    def _person(x):
        return x.split('_')[0]
    x2 = []
    y2 = []
    for f in range(len(x)):
        x2.append(_rem_tif(x[f]))
        y2.append(_rem_mask(y[f]))
    sort_idx = np.empty([len(x),],dtype=int)
    for f in range(len(x)):
        sort_idx[f] = y2.index(x2[f])
    y = [y[i] for i in sort_idx]
    idx = [_person(y[i]) for i in sort_idx]
    return x,y,idx
    

def load_train():
    path = '/home/jason/data/kaggle-uns' # at home
    if not os.path.isdir(path): # at school
        path = '/usr/local/data/jtaylor/Databases/Kaggle-UNS'
    path = os.path.join(path,'train')
    print('Loading training data')
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and f.endswith('.tif')]
    x_files = [f for f in files if not f.endswith('mask.tif')]
    y_files = [f for f in files if f.endswith('mask.tif')]
    
    # sort x_files and y_files so they correspond
    x_files,y_files,idx = _align_files(x_files,y_files)
    
    x = np.empty([len(x_files),1,rows,cols],dtype='float32')
    y = np.empty([len(y_files),1,rows,cols],dtype='float32')
    for f in range(len(x_files)):
        x[f,...] = imresize(image.load_img(os.path.join(path,x_files[f]),grayscale=True),(rows,cols),interp='bilinear')
        y[f,...] = imresize(image.load_img(os.path.join(path,y_files[f]),grayscale=True),[rows,cols],interp='nearest')
    
    # rescale y to 0-1
    y = y/255
    
    return x,y,idx


def load_test():
    path = '/home/jason/data/kaggle-uns' # at home
    if not os.path.isdir(path): # at school
        path = '/usr/local/data/jtaylor/Databases/Kaggle-UNS'
    path = os.path.join(path,'test')
    print('Loading test data')
    x_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and f.endswith('.tif')]
    
    x = np.empty([len(x_files),1,rows,cols],dtype='float32')
    idx = np.empty([len(x_files),],dtype=int)    
    for f in range(len(x_files)):
        x[f,...] = imresize(image.load_img(os.path.join(path,x_files[f]),grayscale=True),(rows,cols),interp='bilinear')
        idx[f] = int(x_files[f].split('.')[0])
        
    return x,idx
    
# Load data
def load_data():
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
        x_train,y_train,idx_train = load_train()
        x_test,idx_test = load_test()
        x_train,x_test = normalize(x_train,x_test)
        
        # Save data for quick reloading
        np.save('x_train',x_train)
        np.save('y_train',y_train)
        np.save('idx_train',idx_train)
        np.save('x_test',x_test)
        np.save('idx_test',idx_test)
    return x_train,y_train,idx_train,x_test,idx_test