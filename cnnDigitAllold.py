# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:22:37 2018

@author: Chandrika
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from keras.callbacks import TensorBoard
import time
from keras.preprocessing.image import ImageDataGenerator


NAME = "Bangla-Digits-{}".format(int(time.time()))
model = Sequential()
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

#model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=10,activation="softmax"))

model.compile(optimizer="adam", loss = "categorical_crossentropy", metrics= ["accuracy"])

train_data = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip = False)
test_data = ImageDataGenerator(rescale=1./255)

training = train_data.flow_from_directory('train',target_size=(32,32), batch_size=50,class_mode='categorical')
test = train_data.flow_from_directory('test',target_size=(32,32), batch_size=50,class_mode='categorical')

#model.fit_generator(training, steps_per_epoch = 80, epochs=50, validation_data=test, validation_steps = 40, callbacks=[tensorboard])
model.fit_generator(training, epochs=40, validation_data=test, callbacks=[tensorboard])

