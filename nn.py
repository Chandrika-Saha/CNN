# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 11:59:22 2018

@author: Arnab
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:46:34 2018

@author: Chandrika
"""


import tensorflow as tf
import numpy as np
from DataSetGenerator import DataSetGenerator


n_nodes_hl1 = 1000
n_nodes_hl2 = 1000
n_nodes_hl3 = 1000
n_nodes_hl4 = 1000
n_classes = 50
batch_size  = 100

x = tf.placeholder('float',shape=[None, 160,160, 1])
y = tf.placeholder('float')

def neural_network_model(data):
    data = tf.reshape(data, [-1,160*160])
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([25600,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']),hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    
    output = tf.matmul(l4,output_layer['weights'])+output_layer['biases']
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epoch = 50
    dg = DataSetGenerator("G:\Matlab Paper Codes\DataSet\BasicFinalDatabase\Train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epoch):
            batches = dg.get_mini_batches(batch_size ,(160,160), allchannel=False)
            epoch_loss = 0
            for imgs ,labels in batches:
                imgs=np.divide(imgs, 255)
                _,acc,c = sess.run([optimizer,accuracy,cost],feed_dict={x: imgs, y: labels})
                epoch_loss += c
            print('Epoch: ',epoch,'Completed out of :',hm_epoch,'Loss: ',epoch_loss," ACC: ",acc*100)
#            for _ in range(int(mnist.train.num_examples/batch_size)):
#                x_epoch,y_epoch = mnist.train.next_batch(batch_size)
#                _,c = sess.run([optimizer,cost],feed_dict = {x:x_epoch,y:y_epoch})
#                epoch_loss += c
#            print('Epoch: ',epoch,'Completed out of :',hm_epoch,'Loss: ',epoch_loss)
#            
        dg = DataSetGenerator("G:\Matlab Paper Codes\DataSet\BasicFinalDatabase\Test")
        j = 0
        for i in range(20):
            test = dg.get_mini_batches(batch_size ,(160,160),allchannel=False)
            for imgs, labels in test:
                imgs=np.divide(imgs, 255)
                imgs=np.add(imgs, -0.5)
                j=j+1
                correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
                accuracy = tf.reduce_mean(tf.cast(correct,'float'))*100
                print("Accuracy: ",j," : ", accuracy.eval({x:imgs,y:labels}))

            
train_neural_network(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    