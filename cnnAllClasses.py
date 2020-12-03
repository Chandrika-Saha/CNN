# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 21:58:49 2018

@author: Chandrika
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:22:37 2018

@author: Chandrika
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import TensorBoard

#i = 64

dense_layers = [1, 2]
filter_size = [32, 32, 64, 64, 64, 128, 128]
conv_layers = [5, 6, 7]
for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        train_data = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip = False)
        test_data = ImageDataGenerator(rescale=1./255)

        training = train_data.flow_from_directory('TrainAll',target_size=(64,64), batch_size=32,class_mode='categorical')
        test = test_data.flow_from_directory('TestAll',target_size=(64,64), batch_size=32,class_mode='categorical')

        NAME = "Bangla_ALL-{}-conv-{}-dense-{}".format(conv_layer,dense_layer,int(time.time()))
        print(NAME)
        model = Sequential()
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=0, write_images=True, write_graph=True, update_freq='epoch')
        model.add(Conv2D(32,(3,3),input_shape=(64,64,3),padding="same",activation="relu"))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2,2),padding="same"))
        
        for l in range(conv_layer-1):
            model.add(Conv2D(filter_size[l+1],(3,3),padding="same",activation="relu"))
            model.add(Dropout(0.3))
            model.add(MaxPooling2D(pool_size=(2,2),padding="same"))
        
        model.add(Flatten())
        
        for k in range(dense_layer):
            model.add(Dense(units=filter_size[k+5],activation="relu"))
            model.add(Dropout(0.1))
        model.add(Dense(units=231,activation="softmax"))
        
        model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics= ["accuracy"])
        
        model.fit_generator(training, epochs=40, validation_data=test, callbacks=[tensorboard])
#        if i==256:
#            i=64
#        else:
#            i = i*2
        
