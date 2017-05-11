import json
from model_helper import early_stopping
from model_helper import get_next_batch
from model_helper import load_feature_sets
import os
import tensorflow as tf

train_x, train_y, test_x, test_y = load_feature_sets()

# Parameters
max_epochs = 1e6
batch_size = 100
total_batch = int(len(train_x)/batch_size)+1
display_step = 1
learning_rate = 0.01
nodes_hl1 = 3
nodes_hl2 = 3
x_shape = len(train_x[0])
y_shape = len(train_y[0])
model_path = "../models/mlp.model"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hidden_1_layer = {"f_fum": nodes_hl1,
                  "weight": tf.Variable(tf.random_normal([x_shape, nodes_hl1])),
                  "bias": tf.Variable(tf.random_normal([nodes_hl1]))}

hidden_2_layer = {"f_fum": nodes_hl2,
                  "weight": tf.Variable(tf.random_normal([nodes_hl1, nodes_hl2])),
                  "bias": tf.Variable(tf.random_normal([nodes_hl2]))}

output_layer = {"f_fum": None,
                "weight": tf.Variable(tf.random_normal([nodes_hl2, y_shape])),
                "bias": tf.Variable(tf.random_normal([y_shape]))}

def model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer["weight"]), hidden_1_layer["bias"])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weight"]), hidden_2_layer["bias"])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer["weight"]) + output_layer["bias"]
    return output

pred = model(x)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
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
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
            ptr += batch_size
        if (epoch+1) % display_step == 0:
            print("Epoch ", epoch, ": cost=", "{:.6f}".format(avg_cost), end='', sep='')
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        res = accuracy.eval({x: test_x, y: test_y})
        print(", accuracy=", res, sep='')
        saver.save(sess, model_path)
        return res

early_stopping(train, max_epochs, model_path)