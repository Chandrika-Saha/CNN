# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:44:52 2018

@author: Chandrika
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import TensorBoard

i = 64

dense_layers = [1, 2]
filter_size = [32, 32, 64, 64, 128, 128, 256]
conv_layers = [5, 6, 7]
for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        train_data = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip = False)
        test_data = ImageDataGenerator(rescale=1./255)

        training = train_data.flow_from_directory('trainB',target_size=(i,i), batch_size=50,class_mode='categorical')
        test = train_data.flow_from_directory('testB',target_size=(i,i), batch_size=50,class_mode='categorical')

        NAME = "Bangla_Basic-{}-conv-{}-dense-{}".format(conv_layer,dense_layer,int(time.time()))
        print(NAME)
        model = Sequential()
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
        model.add(Conv2D(32,(3,3),input_shape=(i,i,3),padding="same",activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        
        for l in range(conv_layer-1):
            model.add(Conv2D(filter_size[l+1],(3,3),padding="same",activation="relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))
        
        model.add(Flatten())
        
        for k in range(dense_layer):
            model.add(Dense(units=filter_size[k+5],activation="relu"))
            
        model.add(Dense(units=50,activation="softmax"))
        
        model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics= ["accuracy"])
        
        model.fit_generator(training, epochs=50, validation_data=test, callbacks=[tensorboard])
        if i==256:
            i=64
        else:
            i = i*2
        
