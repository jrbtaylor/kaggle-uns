# -*- coding: utf-8 -*-
"""
Created on Tue May 31 22:20:56 2016

@author: jason
"""

import numpy as np
np.random.seed(1)  # for reproducibility

# Options
opt = {'rows':105, 'cols':145}

# Load data
import data
x,y = data.load(opt)

# Define model
import model
cnn = model.init(opt)
cnn.compile(loss='categorical_crossentropy',optimizer='adadelta')

# Train
batch_size = 32
nb_epoch = 100
history = cnn.fit(x,y,batch_size=batch_size,nb_epoch=nb_epoch,verbose=1,validation_split=0.04,shuffle=True)


