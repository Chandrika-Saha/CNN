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
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import csv
from keras.utils.vis_utils import plot_model
from keras.callbacks import History
from matplotlib import pyplot as plt


i = 112

dense_layers = [2]
filter_size = [32, 32, 64, 64, 128, 128, 256]
conv_layers = [6]

train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

training = train_data.flow_from_directory('train',target_size=(i,i), batch_size=128,class_mode='categorical')
test = test_data.flow_from_directory('test',target_size=(i,i), batch_size=128,class_mode='categorical')

NAME = "Bangla_Digit_Final-{}-conv-{}-dense-{}".format(conv_layers[0],dense_layers[0],int(time.time()))
print(NAME)
model = Sequential()
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=0, write_images=True, write_graph=True, update_freq='epoch')

model.add(Conv2D(32,(3,3),input_shape=(i,i,3),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

model.add(Conv2D(32,(3,3),padding="same",activation="relu"))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2), padding="valid"))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

model.add(Conv2D(64,(3,3),padding="same",activation="relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2), padding="valid"))

model.add(Conv2D(128,(3,3),padding="same",activation="relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

model.add(Conv2D(512,(3,3),padding="same",activation="relu"))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

model.add(Flatten())

model.add(Dense(1024,activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(512,activation="relu"))
model.add(Dense(units=10,activation="softmax"))

opt= Adam(lr=.0005)
model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics= ["accuracy"])

history = History()
model.fit_generator(training, epochs=100, validation_data=test, callbacks=[tensorboard,history])

print(history.history.keys())
plot_model(model, to_file='DigitFinal.png')
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


start = time.time()
testp = test_data.flow_from_directory('test',target_size=(i,i), batch_size=64,class_mode='categorical',shuffle=False)        
start1 = time.time()
t_pred = model.predict_generator(testp)
end = time.time()
pred = np.argmax(t_pred, axis=1)
print(pred)
print('Confusion Matrix')
conf = confusion_matrix(testp.classes, pred)
print(conf)
print('Classification Report')
target_names = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9']

f1 = classification_report(testp.classes, pred, target_names=target_names)
print(f1)



plt.imshow(conf, cmap=plt.cm.Blues)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.xticks([], [])
plt.yticks([], [])
plt.title('Confusion matrix ')
plt.colorbar()
plt.show()

f = open("DigitFinalConf.csv",'w',newline ='')
a = []
for i in range (0,10):
   print("\n")
   for j in range(0,10):
       a.append(conf[i][j])
       print(conf[i][j], end=' ')
       csv.writer(f).writerow(a)
       a = []
f.close()


ff = open("DigitFinalReport.txt",'w',newline ='')
ff.write(f1)
ff.close()
print("\nDirectory and predict: ",end-start)
print("Predict: ",end-start1)
model.save('digit.h5')
# =============================================================================
# testp = test_data.flow_from_directory('test',target_size=(i,i), batch_size=128,class_mode='categorical',shuffle=False)        
# t_pred = model.predict_generator(testp)
# pred = np.argmax(t_pred, axis=1)
# print(pred)
# print('Confusion Matrix')
# conf = confusion_matrix(testp.classes, pred)
# print(conf)
# np.savetxt('Digit_Conf.txt',conf, delimiter="\n")
# print('Classification Report')
# target_names = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9']
# f1 = classification_report(testp.classes, pred, target_names=target_names)
# print(f1)
# np.savetxt('Digit_Conf.txt',f1, delimiter="\n")
# =============================================================================

