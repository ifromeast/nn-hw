import numpy as np
from math import sin, exp, pi, cos
import tensorflow as tf
import pickle as pkl
import utils
import matplotlib.pyplot as plt

length = 500
y = np.zeros([length])
y[0] = 1
y[1] = 1
for i in range(length):
    y[i] = 0.3 * y[i - 1] + 0.6 * y[i - 2] + 0.6 * sin(
        pi * cos(2 * pi * i / 250)) + 0.9 * exp(-y[i - 1] ** 2)

x = np.linspace(1, length, length)

x = np.transpose([x])
y = np.transpose([y])

batch_size = 32
lr = 1e-3
steps = 1000
model_path = 'model'

x_net = tf.placeholder(tf.float32)
y_net = tf.placeholder(tf.float32)
w = tf.Variable(tf.truncated_normal([1, 10]))
b = tf.Variable(tf.zeros([10]))
w2 = tf.Variable(tf.truncated_normal([10, 1]))
b2 = tf.Variable(tf.zeros([1]))
h = tf.nn.sigmoid(tf.matmul(x_net, w) + b)
predict = tf.nn.relu(tf.matmul(h, w2) + b2)

loss = tf.losses.mean_squared_error(y_net, predict)
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batches = utils.batch_generator([x, y], batch_size)

    for step in range(steps):
        xb, yb = batches.next()
        sess.run(train_op, feed_dict={x_net: xb, y_net: yb})
        if step % 20 == 0:
            ls = sess.run(loss, feed_dict={x_net: x, y_net: y})
            print(step, ": ", ls)

    with open(model_path, 'wb') as f_dump:
        var_map = dict()
        var_map['w1'] = sess.run(w)
        var_map['w2'] = sess.run(w2)
        var_map['b1'] = sess.run(b)
        var_map['b2'] = sess.run(b2)
        pkl.dump(var_map, f_dump)
