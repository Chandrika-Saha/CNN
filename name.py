# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:39:59 2018

@author: Arnab
"""

#%%
import time
i = 64

dense_layers = [1, 2]
filter_size = [32, 32, 64, 64, 128, 128, 256]
conv_layers = [5, 6, 7]
for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        NAME = "Bangla_Digit-{}-conv-{}-dense-{}".format(conv_layer,dense_layer,int(time.time()))
        print(NAME)
        print(i)
        if i==256:
            i=64
        else:
            i = i*2
    