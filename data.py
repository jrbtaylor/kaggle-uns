# -*- coding: utf-8 -*-
"""
Loads Kaggle Ultrasound Nerve Segmentation data

@author: jason
"""

import os
import numpy as np
from keras.preprocessing import image
from scipy.misc import imresize

def preprocess_x(x):
    x -= np.min(x)
    x = x/np.std(x)
    return x
    
def preprocess_y(x):
    y = np.zeros([x.shape[0],x.shape[1],2])
    for r in range(x.shape[0]):
        for c in range(x.shape[1]):
            if x[r,c]>0.5:
                y[r,c,1] = 1
            else:
                y[r,c,0] = 1
    y = np.reshape(y,[x.shape[0]*x.shape[1],2])
    return y

def load(opt):  
    path = '/home/jason/datasets/Kaggle_UNS/train'
    print('Loading data')
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f)) and f.endswith('.tif')]
    x_files = [f for f in files if not f.endswith('mask.tif')]
    y_files = [f for f in files if f.endswith('mask.tif')]
    
    # input image dimensions
    img_rows, img_cols = opt['rows'], opt['cols'] # 1/4 the original size, upsample results for test
    
    x = np.empty([len(x_files),1,img_rows,img_cols],dtype=np.float32)
    y = np.empty([len(y_files),img_rows*img_cols,2],dtype=np.float32)
    
    for f in range(len(x_files)):
        x[f,...] = preprocess_x(imresize(image.load_img(os.path.join(path,x_files[f]),grayscale=True),(img_rows,img_cols),interp='bilinear'))
        y[f,...] = preprocess_y(imresize(image.load_img(os.path.join(path,y_files[f]),grayscale=True),[img_rows,img_cols],interp='nearest'))

    return x,y

