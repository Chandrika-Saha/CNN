# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:44:52 2018

@author: Chandrika
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix

i = 128

dense_layers = [2]
filter_size = [32, 32, 64, 64, 128, 512, 1024]
conv_layers = [6, 7]
for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        train_data = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip = False)
        test_data = ImageDataGenerator(rescale=1./255)

        training = train_data.flow_from_directory('trainC',target_size=(i,i), batch_size=50,class_mode='categorical')
        test = test_data.flow_from_directory('testC',target_size=(i,i), batch_size=15,class_mode='categorical')

        NAME = "Bangla_Compound-{}-conv-{}-dense-{}".format(conv_layer,dense_layer,int(time.time()))
        print(NAME)
        model = Sequential()
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=0, write_images=True, write_graph=True, update_freq='epoch')
        model.add(Conv2D(32,(3,3),input_shape=(i,i,3),padding="same",activation="relu"))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2,2)))
       
        
        for l in range(conv_layer-1):
            model.add(Conv2D(filter_size[l+1],(3,3),padding="same",activation="relu"))
            model.add(Dropout(0.25))
            model.add(MaxPooling2D(pool_size=(2,2)))
            
        
        model.add(Flatten())
        
        for k in range(dense_layer):
            model.add(Dense(units=filter_size[k+5],activation="relu"))
            model.add(Dropout(0.25))
            
        model.add(Dense(units=171,activation="softmax"))
        
        model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics= ["accuracy"])
        
        model.fit_generator(training, epochs=25, validation_data=test, callbacks=[tensorboard])
        if i==512:
            i=256
        else:
            i = i+50
            
        t_pred = model.predict_generator(test)
        pred = np.argmax(t_pred, axis=1)
        print(pred)
        print('Confusion Matrix')
        print(confusion_matrix(test.classes, pred))
        print('Classification Report')
        target_names = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9']
        print(classification_report(test.classes, pred, target_names=target_names))
        
