#!/usr/bin/env python3
import json
from model_helper import early_stopping
from model_helper import get_next_batch
from model_helper import load_feature_sets
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import rnn

MAX_EPOCHS = 1e6
BATCH_SIZE = 100
DISPLAY_STEP = 1
LEARNING_RATE = 0.01
CHUNKS = 1
RNN_SIZE = 24
MODEL_PATH = "../models/rnn.model"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model(data, x_shape, y_shape):
    """
    Model definition.
    """
    layer = {'weights': tf.Variable(tf.random_normal([RNN_SIZE, y_shape])), 'biases': tf.Variable(tf.random_normal([y_shape]))}
    data = tf.transpose(data, [1,0,2])
    data = tf.reshape(data, [-1, x_shape])
    data = tf.split(data, CHUNKS, 0)
    lstm_cell = rnn.BasicLSTMCell(RNN_SIZE, state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output


"""
Model initialization.
"""
train_x, train_y, test_x, test_y = load_feature_sets()

total_batch = int(len(train_x)/BATCH_SIZE)
x_shape = len(train_x[0]) # Chunk size
y_shape = len(train_y[0])

# tf Graph Input
x = tf.placeholder(tf.float32, [None, CHUNKS, x_shape])
y = tf.placeholder(tf.float32)

pred = model(x, x_shape, y_shape)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
tf.add_to_collection('vars', pred)
tf.add_to_collection('vars', x)
tf.add_to_collection('vars', y)


def train(epoch):
    """
    Train the model.
    """
    with tf.Session() as sess:
        sess.run(init)
        if epoch != 1:
            saver.restore(sess, MODEL_PATH)
        avg_cost = 0.
        ptr = 0
        for i in range(total_batch):
            batch_x, batch_y = get_next_batch(ptr, BATCH_SIZE, train_x, train_y)
            batch_x = batch_x.reshape((BATCH_SIZE, CHUNKS, x_shape))
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
            ptr += BATCH_SIZE
        if (epoch+1) % DISPLAY_STEP == 0:
            print("Epoch ", epoch, ": cost=", "{:.6f}".format(avg_cost), end='', sep='')
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        res = accuracy.eval({x: np.array(test_x).reshape((-1, CHUNKS, x_shape)), y: test_y})
        print(", accuracy=", res, sep='')
        saver.save(sess, MODEL_PATH)
        return res


early_stopping(train, MAX_EPOCHS, MODEL_PATH)