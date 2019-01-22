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
NODES_HL1 = 3
NODES_HL2 = 3
MODEL_PATH = "../models/mlp.model"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model(data, hidden_1_layer, hidden_2_layer, output_layer):
    """
    Model definition.
    """
    l1 = tf.add(tf.matmul(data, hidden_1_layer["weight"]), hidden_1_layer["bias"])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weight"]), hidden_2_layer["bias"])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer["weight"]) + output_layer["bias"]
    return output


"""
Model initialization.
"""
train_x, train_y, test_x, test_y = load_feature_sets()

total_batch = int(len(train_x)/BATCH_SIZE)+1
x_shape = len(train_x[0])
y_shape = len(train_y[0])

# tf Graph Input
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# Layers
hidden_1_layer = {"f_fum": NODES_HL1,
              "weight": tf.Variable(tf.random_normal([x_shape, NODES_HL1])),
              "bias": tf.Variable(tf.random_normal([NODES_HL1]))}
hidden_2_layer = {"f_fum": NODES_HL2,
              "weight": tf.Variable(tf.random_normal([NODES_HL1, NODES_HL2])),
              "bias": tf.Variable(tf.random_normal([NODES_HL2]))}
output_layer = {"f_fum": None,
            "weight": tf.Variable(tf.random_normal([NODES_HL2, y_shape])),
            "bias": tf.Variable(tf.random_normal([y_shape]))}

pred = model(x, hidden_1_layer, hidden_2_layer, output_layer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
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
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
            ptr += BATCH_SIZE
        if (epoch+1) % DISPLAY_STEP == 0:
            print("Epoch ", epoch, ": cost=", "{:.6f}".format(avg_cost), end='', sep='')
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        res = accuracy.eval({x: test_x, y: test_y})
        print(", accuracy=", res, sep='')
        saver.save(sess, MODEL_PATH)
        return res


early_stopping(train, MAX_EPOCHS, MODEL_PATH)