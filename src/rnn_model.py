import json
from model_helper import early_stopping
from model_helper import get_next_batch
from model_helper import load_feature_sets
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import rnn

train_x, train_y, test_x, test_y = load_feature_sets()

# Parameters
max_epochs = 1e6
batch_size = 100
total_batch = int(len(train_x)/batch_size)
display_step = 1
learning_rate = 0.01
chunks = 1
rnn_size = 24
x_shape = len(train_x[0]) # Chunk size
y_shape = len(train_y[0])
model_path = "../models/rnn.model"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.placeholder(tf.float32, [None, chunks, x_shape])
y = tf.placeholder(tf.float32)

def model(data):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, y_shape])), 'biases': tf.Variable(tf.random_normal([y_shape]))}
    data = tf.transpose(data, [1,0,2])
    data = tf.reshape(data, [-1, x_shape])
    data = tf.split(data, chunks, 0)
    lstm_cell = rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, data, dtype=tf.float32)
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']
    return output

pred = model(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
tf.add_to_collection('vars', pred)
tf.add_to_collection('vars', x)
tf.add_to_collection('vars', y)

def train(epoch):
    with tf.Session() as sess:
        sess.run(init)
        if epoch != 1:
            saver.restore(sess, model_path)
        avg_cost = 0.
        ptr = 0
        for i in range(total_batch):
            batch_x, batch_y = get_next_batch(ptr, batch_size, train_x, train_y)
            batch_x = batch_x.reshape((batch_size, chunks, x_shape))
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
            ptr += batch_size
        if (epoch+1) % display_step == 0:
            print("Epoch ", epoch, ": cost=", "{:.6f}".format(avg_cost), end='', sep='')
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        res = accuracy.eval({x: np.array(test_x).reshape((-1, chunks, x_shape)), y: test_y})
        print(", accuracy=", res, sep='')
        saver.save(sess, model_path)
        return res

early_stopping(train, max_epochs, model_path)