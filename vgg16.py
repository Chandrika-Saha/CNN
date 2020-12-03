

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import History
from matplotlib import pyplot as plt




model = Sequential()
model_new = VGG16(include_top=False, weights=None, input_shape=(224,224,3))
model.add(model_new)


print(model.summary())


NAME = "Bangla_Compound_Final-{}-conv-{}-dense-{}".format("conv_layers","dense_layers",int(time.time()))
print(NAME)
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=0, write_images=True, write_graph=True, update_freq='epoch')


train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

training = train_data.flow_from_directory('trainC',target_size=(224,224), batch_size=16,class_mode='categorical')
test = test_data.flow_from_directory('testC',target_size=(224,224), batch_size=16,class_mode='categorical')



model.add(Flatten())

model.add(Dense(1024,activation="relu"))

model.add(Dense(512,activation="relu"))
model.add(Dense(units=171,activation="softmax"))

history = History()
opt= Adam(lr=.00001)
model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics= ["accuracy"])

model.fit_generator(training, epochs=30, validation_data=test, callbacks=[tensorboard,history])


print(history.history.keys())
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

testp = test_data.flow_from_directory('testC',target_size=(224,224), batch_size=16,class_mode='categorical',shuffle=False)        
t_pred = model.predict_generator(testp)
pred = np.argmax(t_pred, axis=1)
print(pred)
print('Confusion Matrix')
conf = confusion_matrix(testp.classes, pred)
print(conf)
print('Classification Report')
target_names = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9']
for k in range(10,171):
    k1=k
    target_names.append(str(k1))
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

print(model.summary())   

