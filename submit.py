# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:33:13 2016

@author: jason
"""

import numpy as np

def run_length_enc(y):
    from itertools import chain
    y = np.where(y)[0]
    if len(y) < 1:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])
    
def prep(y):
    from scipy.misc import imresize
    rows_out, cols_out = 420, 580
    y = imresize(np.squeeze(y),[rows_out,cols_out],interp='bilinear')>0.5
    y = np.transpose(y)
    y = np.reshape(y,[rows_out*cols_out])
    return y
    
    
def final(y,idx):
    argsort = np.argsort(idx)
    y = y[argsort,:,:]
    idx = idx[argsort]
    total = idx.shape[0]
        
    rle = []
    for i in range(total):
        rle.append(run_length_enc(prep(y[i,:,:])))
        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(idx[i]) + ',' + rle[i]
            f.write(s + '\n')