# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:22:37 2018

@author: Chandrika
"""
from keras.optimizers import Adam
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from keras.callbacks import TensorBoard
import time
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

NAME = "Bangla-Digits-{}".format(int(time.time()))
model = Sequential()

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=0, write_images=True, write_graph=True, update_freq='epoch')
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=10,activation="softmax"))

opt= Adam(lr=.001)
model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics= ["accuracy"])

train_data = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip = False)
test_data = ImageDataGenerator(rescale=1./255)

training = train_data.flow_from_directory('train',target_size=(32,32), batch_size=50,class_mode='categorical')
test = test_data.flow_from_directory('test',target_size=(32,32), batch_size=50, shuffle=False, class_mode='categorical')


#model.fit_generator(training, steps_per_epoch = 80, epochs=50, validation_data=test, validation_steps = 40, callbacks=[tensorboard])
model.fit_generator(training, steps_per_epoch = 80, epochs=25, validation_data=test, validation_steps = 40, callbacks=[tensorboard])


#confusion matrix

t_pred = model.predict_generator(test)
pred = np.argmax(t_pred, axis=1)
print(pred)
print('Confusion Matrix')
conf = confusion_matrix(test.classes, pred)
print(conf)
np.savetxt('Digit_Conf.txt',conf, delimiter="\n")
print('Classification Report')
target_names = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9']
f1 = classification_report(test.classes, pred, target_names=target_names)
print(f1)
np.savetxt('Digit_Conf.txt',f1, delimiter="\n")