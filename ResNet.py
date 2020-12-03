
# coding: utf-8

# In[1]:


from keras.models import Sequential
import math
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation,Input,Add, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import time
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import History, LearningRateScheduler
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model


# In[2]:


i = 64


# In[3]:


model = Sequential()


# In[4]:


NAME = "Bangla_Compound_DCNN171_lr0.0001-{}".format(int(time.time()))
print(NAME)
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=0, write_images=True, write_graph=True, update_freq='epoch')


# In[5]:


train_data = ImageDataGenerator(rescale=1./255)
test_data = ImageDataGenerator(rescale=1./255)

training = train_data.flow_from_directory('trainC',target_size=(i,i), batch_size=64,class_mode='categorical')
test = test_data.flow_from_directory('testC',target_size=(i,i), batch_size=64,class_mode='categorical')
epochs = 50
def step_decay(epochs):
    initial_lrate = 0.0009
    drop = 0.5
    epochs_drop = 5.0
    lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epochs)/epochs_drop))
    return lrate

lrate=LearningRateScheduler(step_decay)


# In[6]:


input_shape = (i,i,3)
X_input = Input(input_shape)
x = Conv2D(32,(3,3),input_shape=(i,i,3),padding="same")(X_input)
x_short = x
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(32,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Add()([x, x_short])
#x = Dropout(0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(64,(3,3),padding="same")(x)
x_short = x
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(64,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Add()([x, x_short])
#x = Dropout(0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)


x = Conv2D(128,(3,3),padding="same")(x)
x_short = x
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(128,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Add()([x, x_short])
#x = Dropout(0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(256,(3,3),padding="same")(x)
x_short = x
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(256,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(256,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Add()([x, x_short])
#x = Dropout(0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)

x = Conv2D(512,(3,3),padding="same")(x)
x_short = x
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(512,(3,3),padding="same")(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Add()([x, x_short])
#x = Dropout(0.1)(x)
x = MaxPooling2D(pool_size=(2,2))(x)


x = Flatten()(x)

x = Dense(1024,activation="relu")(x)
x = Dropout(0.3)(x)

x = Dense(512,activation="relu")(x)

x = Dense(units=171,activation="softmax")(x)
model = Model(inputs = X_input, outputs = x, name='ResNet50')
print(model.summary())
plot_model(model, to_file='withnormIncreasedLayers.png')


# In[7]:



history = History()
opt= Adam(lr=step_decay(epochs), beta_1=0.9, beta_2=0.999)
model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics= ["accuracy"])
model.fit_generator(training, epochs=50, validation_data=test, callbacks=[tensorboard,history,lrate])


# In[8]:



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


testp = test_data.flow_from_directory('testC',target_size=(i,i), batch_size=64,class_mode='categorical',shuffle=False)        
t_pred = model.predict_generator(testp)
pred = np.argmax(t_pred, axis=1)
print(pred)
print('Confusion Matrix')
conf = confusion_matrix(testp.classes, pred)
print(conf)
print(str(conf))
print('Classification Report')
target_names = ['1', '2', '3', '4', '5','6', '7', '8', '9']
for k in range(10,172):
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


# In[9]:


print(str(conf))


# In[10]:


for i in range (0,171):
    print("\n")
    for j in range(0,171):
        print(conf[i][j], end=' ')
        
plt.plot(lrate)
plt.title('Learning rate decay')
plt.ylabel('learning rate')
plt.xlabel('epoch')
plt.legend(["learning rate"], loc='upper left')
plt.show()

