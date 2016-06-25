# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:48:44 2016

@author: jason
"""

from scipy import ndimage
import numpy as np


def _binary(y):
    return y>0.5
    
    
def _fill(y):
    return ndimage.binary_fill_holes(y)


# remove all but the largest connected region of ones
# only one nerve segmentation per image
def _one_region(y):
    # connected component analysis
    labels, nb_labels = ndimage.label(y)
    sizes = ndimage.sum(y,labels,range(nb_labels+1))
    
    # remove all but the biggest region
    remove = sizes<max(sizes)
    remove = remove[labels]
    y[remove] = 0
    
    return y


# remove predicted regions smaller than 80% of the smallest in training set 
# probably a false detection...
def _remove_small(y,y_train):
    # find smallest segmentation mask in training set
    sizes = np.sum(y_train,axis=(3,2))
    smallest = np.min(sizes[sizes>0])
    
    # connected component analysis
    labels, nb_labels = ndimage.label(y)
    sizes = ndimage.sum(y,labels,range(nb_labels+1))
    
    # remove if smaller than 0.8*smallest
    remove = sizes<0.8*smallest
    remove = remove[labels]
    y[remove] = 0
    
    return y
    
    
def final(y,y_train):
    y = _binary(y)
    for i in range(y.shape[0]):
        y[i] = _fill(y[i])
        y[i] = _one_region(y[i])
        y[i] = _remove_small(y[i],y_train)
    return y