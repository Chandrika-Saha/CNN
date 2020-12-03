# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 12:34:23 2018

@author: Chandrika
"""
import tensorflow as tf

x1 = tf.constant(100)
x2 = tf.constant(100)
s = tf.constant("hello tf")
result = tf.multiply(x1,x2)

#sess = tf.Session()
#
#print(sess.run(result))
#print(sess.run(s))
#sess.close()

with tf.Session() as sess:
    print(sess.run(result))