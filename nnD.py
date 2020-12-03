# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 19:15:18 2018

@author: Arnab
"""

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


n_nodes_hl1 = 5000
n_nodes_hl2 = 5000
n_nodes_hl3 = 5000
n_nodes_hl4 = 5000
n_nodes_hl11 = 5000
n_nodes_hl22 = 5000
n_nodes_hl33 = 5000
n_nodes_hl44 = 5000
n_classes = 10
batch_size  = 100

x = tf.placeholder('float',shape=[None, 32,32, 1])
y = tf.placeholder('float')

def neural_network_model(data):
    data = tf.reshape(data, [-1,32*32])
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([1024,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4])),'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}
    hidden_11_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4,n_nodes_hl11])),'biases':tf.Variable(tf.random_normal([n_nodes_hl11]))}
    hidden_22_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl11,n_nodes_hl22])),'biases':tf.Variable(tf.random_normal([n_nodes_hl22]))}
    hidden_33_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl22,n_nodes_hl33])),'biases':tf.Variable(tf.random_normal([n_nodes_hl33]))}
    hidden_44_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl33,n_nodes_hl44])),'biases':tf.Variable(tf.random_normal([n_nodes_hl44]))}
    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl44,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']),hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    
    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']),hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weights']),hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)
    
    l11 = tf.add(tf.matmul(l4,hidden_11_layer['weights']),hidden_11_layer['biases'])
    l11 = tf.nn.relu(l11)
    
    l22 = tf.add(tf.matmul(l11,hidden_22_layer['weights']),hidden_22_layer['biases'])
    l22 = tf.nn.relu(l22)
    
    l33 = tf.add(tf.matmul(l22,hidden_33_layer['weights']),hidden_33_layer['biases'])
    l33 = tf.nn.relu(l33)
    
    l44 = tf.add(tf.matmul(l33,hidden_44_layer['weights']),hidden_44_layer['biases'])
    l44 = tf.nn.relu(l44)
    output = tf.matmul(l44,output_layer['weights'])+output_layer['biases']
    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epoch = 30
    dg = DataSetGenerator(".\Train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epoch):
            batches = dg.get_mini_batches(batch_size ,(32,32), allchannel=False)
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
        dg = DataSetGenerator(".\Test")
        j = 0
        for i in range(10):
            test = dg.get_mini_batches(200 ,(32,32),allchannel=False)
            for imgs, labels in test:
                imgs=np.divide(imgs, 255)
                imgs=np.add(imgs, -0.5)
                j=j+1
                correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
                accuracy = tf.reduce_mean(tf.cast(correct,'float'))*100
                print("Accuracy: ",j," : ", accuracy.eval({x:imgs,y:labels}))

            
train_neural_network(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    