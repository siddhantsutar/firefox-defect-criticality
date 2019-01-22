#!/usr/bin/env python3
import json
from model_helper import early_stopping
from model_helper import get_next_batch
from model_helper import load_feature_sets
import os
import tensorflow as tf

MAX_EPOCHS = 1e6
BATCH_SIZE = 100
DISPLAY_STEP = 1
LEARNING_RATE = 0.01
MODEL_PATH = "../models/logistic.model"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model(x, W, b):
	"""
	Model definition.
	"""
	return tf.nn.softmax(tf.matmul(x, W) + b) # Softmax


"""
Model initialization.
"""
train_x, train_y, test_x, test_y = load_feature_sets()

total_batch = int(len(train_x)/BATCH_SIZE)+1
x_shape = len(train_x[0])
y_shape = len(train_y[0])

# tf Graph Input
x = tf.placeholder(tf.float32, [None, x_shape])
y = tf.placeholder(tf.float32, [None, y_shape])

# Set model weights
W = tf.Variable(tf.zeros([x_shape, y_shape])) 
b = tf.Variable(tf.zeros([y_shape]))

pred = model(x, W, b) # Construct model
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1)) # Minimize error using cross entropy
optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost) # Gradient Descent
init = tf.global_variables_initializer() # Initializing the variables
saver = tf.train.Saver()
tf.add_to_collection('vars', pred)
tf.add_to_collection('vars', x)
tf.add_to_collection('vars', y)


def train(epoch):
	"""
	Train the model.
	"""
	with tf.Session() as sess: # Launch the graph
		sess.run(init)
		if epoch != 1:
			saver.restore(sess, MODEL_PATH)
		avg_cost = 0.
		ptr = 0
		for i in range(total_batch): # Loop over all batches
			batch_x, batch_y = get_next_batch(ptr, BATCH_SIZE, train_x, train_y)
			_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y}) # Run optimization op (backprop) and cost op (to get loss value)
			avg_cost += c / total_batch # Compute average loss
			ptr += BATCH_SIZE
		if (epoch+1) % DISPLAY_STEP == 0:
			print("Epoch ", epoch, ": cost=", "{:.6f}".format(avg_cost), end='', sep='')
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)) # Test model
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Calculate accuracy
		res = accuracy.eval({x: test_x, y: test_y})
		print(", accuracy=", res, sep='')
		saver.save(sess, MODEL_PATH)
		return res


early_stopping(train, MAX_EPOCHS, MODEL_PATH)