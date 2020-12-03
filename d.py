# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 13:54:05 2018

@author: Chandrika
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:58:33 2018

@author: Chandrika
"""


import tensorflow as tf
import numpy as np
import csv
from DataSetGenerator import DataSetGenerator
#from tensorflow.examples.tutorials.mnist import input_data
#
#mnist = input_data.read_data_sets("/data",one_hot = True)

n_classes = 10
batch_size  = 2000

x = tf.placeholder('float',shape=[None, 32,32, 1])
y = tf.placeholder('float')
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    
    
def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_conv3':tf.Variable(tf.random_normal([3,3,64,128])),
               'W_fc':tf.Variable(tf.random_normal([4*4*128,1024])),
               'out':tf.Variable(tf.random_normal([1024,n_classes]))}
    
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_conv3':tf.Variable(tf.random_normal([128])),
              'b_fc':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([n_classes]))}
    
    x_img = tf.reshape(x,shape=[-1,32,32,1])
    
    conv1 = tf.nn.relu(conv2d(x_img,weights['W_conv1'])+biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2'])+biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    
    conv3 = tf.nn.relu(conv2d(conv2,weights['W_conv3'])+biases['b_conv3'])
    conv3 = maxpool2d(conv3)
    
    fc = tf.reshape(conv3, [-1,4*4*128])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc,keep_rate)
    
    output = tf.matmul(fc, weights['out'])+ biases['out']
    
    return output

def train_neural_network(x):
    f = open('R600','w')
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epoch = 600
    dg = DataSetGenerator("./train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
            
        for epoch in range(hm_epoch):
            batches = dg.get_mini_batches(batch_size ,(32,32), allchannel=False)
            epoch_loss = 0
            k = 0
            for imgs ,labels in batches:
                k=k+1
                #imgs = np.divide(imgs,255)
                #print("inner for: ",k)
                _,acc,c = sess.run([optimizer,accuracy,cost],feed_dict={x: imgs, y: labels})
                epoch_loss += c
            #print('Epoch: ',epoch,'Completed out of :',hm_epoch,'Loss: ',epoch_loss," ACC: ",acc*100)
            f.write(str(acc*100)+'\n')
#            for _ in range(int(mnist.train.num_examples/batch_size)):
#                x_epoch,y_epoch = mnist.train.next_batch(batch_size)
#                _,c = sess.run([optimizer,cost],feed_dict = {x:x_epoch,y:y_epoch})
#                epoch_loss += c
#            print('Epoch: ',epoch,'Completed out of :',hm_epoch,'Loss: ',epoch_loss)
#            
        dg = DataSetGenerator("./test")
        for cnt in range(10):
            test = dg.get_mini_batches(2000,(32,32),allchannel=False)
            i = 0
            for imgs, labels in test:
                i=i+1
               # imgs = np.divide(imgs,255)
                correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
                accuracy = tf.reduce_mean(tf.cast(correct,'float'))*100
                print("Accuracy: ", accuracy.eval({x:imgs,y:labels}))
    #        accuracy = tf.divide(accuracy,i)
    #        print("Avg ACC:" ,accuracy.eval())
            
train_neural_network(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    