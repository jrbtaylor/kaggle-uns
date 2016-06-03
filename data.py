# -*- coding: utf-8 -*-
"""
Loads Kaggle Ultrasound Nerve Segmentation data

@author: jason
"""

import os
import numpy as np
from keras.preprocessing import image
from scipy.misc import imresize

rows = 64
cols = 80

def normalize(x_train,x_test):
    print('Normalizing data')
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train-mean)/std
    x_test = (x_test-mean)/std
    return x_train,x_test

def load_train():
    path = '/home/jason/datasets/Kaggle_UNS' # at home
    if not os.path.isdir(path): # at school
        path = '/usr/local/data/jtaylor/Databases/Kaggle-UNS'
    path = os.path.join(path,'train')
    print('Loading training data')
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and f.endswith('.tif')]
    x_files = [f for f in files if not f.endswith('mask.tif')]
    y_files = [f for f in files if f.endswith('mask.tif')]
    
    x = np.empty([len(x_files),1,rows,cols],dtype=np.float32)
    y = np.empty([len(y_files),1,rows,cols],dtype=np.float32)
    for f in range(len(x_files)):
        x[f,...] = imresize(image.load_img(os.path.join(path,x_files[f]),grayscale=True),(rows,cols),interp='bilinear')
        y[f,...] = imresize(image.load_img(os.path.join(path,y_files[f]),grayscale=True),[rows,cols],interp='nearest')

    return x,y

def load_test():
    path = '/home/jason/datasets/Kaggle_UNS' # at home
    if not os.path.isdir(path): # at school
        path = '/usr/local/data/jtaylor/Databases/Kaggle-UNS'
    path = os.path.join(path,'test')
    print('Loading test data')
    x_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and f.endswith('.tif')]
    
    x = np.empty([len(x_files),1,rows,cols],dtype=np.float32)
    idx = np.empty([len(x_files),],dtype=np.int32)    
    for f in range(len(x_files)):
        x[f,...] = imresize(image.load_img(os.path.join(path,x_files[f]),grayscale=True),(rows,cols),interp='bilinear')
        idx[f] = int(x_files[f].split('.')[0])
        
    return x,idx