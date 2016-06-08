# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 17:14:20 2016

modified from: https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
to allow for augmentation of the segmentation mask along with the image

@author: jason
"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import scipy.ndimage as ndi
import threading

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x
    
class Generator(object):
    ''' Generate minibatches w/ data augmentation
    includes:
        - multiplicative speckle noise
        - horizontal flipping
        - rotations 
        - zooming (scaling up and cropping)
        - shearing
    '''
    def __init__(self,
                 horizontal_flip = False,
                 vertical_flip = False,
                 rotation_range = 0.,
                 zoom_range = 0.,
                 shear_range = 0.):
        self.__dict__.update(locals())
        self.zoom_range = [1,1+zoom_range]
    
    def flow(self, x, y, batch_size=32, shuffle=True, seed=None):
        return NumpyArrayIterator(
            x, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed)
    
    def random_transform(self,x,y):
        rotation = np.pi/180*np.random.uniform(-self.rotation_range,self.rotation_range)
        rotation = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                             [np.sin(rotation), np.cos(rotation), 0],
                             [0, 0, 1]])
        shear = np.random.uniform(-self.shear_range,self.shear_range)
        shear = np.array([[1, -np.sin(shear),0],
                          [0,  np.cos(shear),0],
                          [0, 0, 1]])
        zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom = np.array([[zx, 0, 0],
                         [0, zy, 0],
                         [0, 0, 1]])
        transform = np.dot(np.dot(rotation,shear),zoom)
        
        h,w = x.shape[1],x.shape[2]
        transform = transform_matrix_offset_center(transform,h,w)
        x = apply_transform(x,transform)
        y = apply_transform(y,transform)
        
        y = np.float32(y>0.5) # fix labels back to {0,1}
        
        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = x[:,::-1]
                y = y[:,::-1]
                
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = x[::-1,:]
                y = y[::-1,:]
                
        return x,y
        

class Iterator(object):
    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)
        

class NumpyArrayIterator(Iterator):
    def __init__(self, X, Y, image_data_generator,
                 batch_size=32, shuffle=True, seed=None):
        self.X = X
        self.Y = Y
        self.image_data_generator = image_data_generator
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        batch_y = np.zeros(tuple([current_batch_size] + list(self.Y.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            y = self.Y[j]
            x,y = self.image_data_generator.random_transform(x.astype('float32'),y.astype('float32'))
            batch_x[i] = x
            batch_y[i] = y
        return batch_x, batch_y