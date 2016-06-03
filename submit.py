# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:33:13 2016

@author: jason
"""

import numpy as np

def run_length_enc(y):
    from data import rows, cols
    from itertools import chain
    from scipy.misc import imresize
    rows_out, cols_out = 420, 580
    y = y[:,:,1]
    y = np.reshape(y,[y.shape[0],rows,cols])    
    yo = np.empty([y.shape[0],rows_out,cols_out],dtype=np.uint8)
    for i in range(y.shape[0]):
        yo[i,...] = imresize(y[i,...],[rows_out,cols_out],interp='bilinear')>0.5
    yo = np.reshape(yo,[yo.shape[0],rows_out*cols_out])
    y = np.where(yo)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])
    
def final(y,idx):
    argsort = np.argsort(idx)
    y = y[argsort]
    idx = idx[argsort]
    total = idx.shape[0]
        
    rle = []
    for i in range(total):
        rle.append(run_length_enc(y))
        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(idx[i]) + ',' + rle[i]
            f.write(s + '\n')