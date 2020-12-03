# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:22:37 2018

@author: Chandrika
"""

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape=(50,50,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=500,activation="relu"))
model.add(Dense(units=50,activation="softmax"))

model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics= ["accuracy"])

train_data = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip = False)
test_data = ImageDataGenerator(rescale=1./255)

training = train_data.flow_from_directory('trainB',target_size=(50,50), batch_size=30,class_mode='categorical')
test = train_data.flow_from_directory('testB',target_size=(50,50), batch_size=30,class_mode='categorical')

model.fit_generator(training, steps_per_epoch = 400, epochs=30, validation_data=test, validation_steps = 100)

