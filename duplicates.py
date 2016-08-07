# -*- coding: utf-8 -*-
"""
Find (near) duplicates in the dataset and return a weighted average of their
label probabilities

Created on Mon Jul 18 21:15:23 2016

@author: jason
"""

def _dist(x):
    import numpy as np
    import scipy.spatial.distance as sp
    
    # sum the pixels in tiles for a feature vector of each image
    tiles = 8
    drow = int(round(x.shape[2]/tiles))
    dcol = int(round(x.shape[3]/tiles))
    sums = np.empty([x.shape[0],tiles,tiles],dtype='float32')
    for im in range(x.shape[0]):
        for row in range(tiles):
            for col in range(tiles):
                tile = x[im,0,row*drow:(row*drow+drow),col*dcol:(col*dcol+dcol)]
                sums[im,row,col] = np.sum(tile)
    sums = sums.reshape((sums.shape[0],-1))
    
    # calculate pairwise distances between feature vectors
    dist = sp.squareform(sp.pdist(sums,'euclidean'))
    
    return dist


# return a probability for each image to contain a label based on its actual value
# plus a weighted sum of similar image labels
def plabel(x,y,idx):
    import numpy as np
    
    y_prob = np.zeros(y.shape[0])
    
    # loop through each individual
    for i in set(idx):
        setidx = [p for p in range(len(idx)) if int(idx[p])==int(i)]
        dist = _dist(x[setidx])
        
        y_set = y[setidx]
        y_set = np.any(y_set,axis=(1,2,3))
        
        # reverse and normalize the distances for each image in set
        for ii in range(len(setidx)):
            distii = dist[ii]
            distii = np.max(distii)-distii
            distii = np.power(distii,5)
            distii = distii/np.sum(distii)
            # ^now a probability distribution inverse proportional to distance
            
            y_prob[setidx[int(ii)]] = np.dot(distii,y_set)
    
    return y_prob


# return the training set re-using the labels of similar images 
# in cases where no label is given
# if in the top prc% percentile of the closest matches
def dupmask(x,y,idx,prc=50,per_patient=False):
    import numpy as np
    
    if per_patient:
        # loop through each individual
        for i in set(idx):
            setidx = [p for p in range(len(idx)) if int(idx[p])==int(i)]
            
            # calculate distances between images for this patient
            dist = _dist(x[setidx])
            
            # which are missing labels?
            missing = [p for p in range(len(setidx)) if not np.any(y[setidx[p]])]
            notmissing = [p for p in range(len(setidx)) if np.any(y[setidx[p]])]
            
            # find closest match with a label for each missing a label
            min_dist = np.amin(dist[missing,:][:,notmissing],axis=1)
            closest_match = np.argmin(dist[missing,:][:,notmissing],axis=1)
            
            # replace missing labels in (some of?) the closest matches
            threshold = np.percentile(min_dist,prc)
            replace = [setidx[missing[p]] for p in range(len(missing)) if min_dist[p]<threshold]
            replace_with = [setidx[notmissing[closest_match[p]]] for p in range(len(closest_match)) if min_dist[p]<threshold]
            y[replace] = y[replace_with]
            
    else: # calculate the distance matrix for all images (slower but only run once)
        dist = _dist(x)
        
        # which are missing labels
        missing = [p for p in range(x.shape[0]) if not np.any(y[p])]
        notmissing = [p for p in range(x.shape[0]) if np.any(y[p])]
        
        # closest matches
        min_dist = np.amin(dist[missing,:][:,notmissing],axis=1)
        closest_match = np.argmin(dist[missing,:][:,notmissing],axis=1)
        
        # replace
        threshold = np.percentile(min_dist,prc)
        replace = [missing[p] for p in range(len(missing)) if min_dist[p]<threshold]
        replace_with = [notmissing[closest_match[p]] for p in range(len(closest_match)) if min_dist[p]<threshold]
        y[replace] = y[replace_with]
        
    return y
    
    
    
    
    
    
    