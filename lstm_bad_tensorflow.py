import numpy as np
import matplotlib.pyplot as plt
from math import sin, exp, pi, cos
import tensorflow as tf

length = 500
data_y = np.zeros([length])
data_y[0] = 1
data_y[1] = 1
for i in range(length):
    data_y[i] = 0.3 * data_y[i - 1] + 0.6 * data_y[i - 2] + 0.6 * sin(
        pi * cos(2 * pi * i / 250)) + 0.9 * exp(-data_y[i - 1] ** 2)

data_x = np.linspace(1, length, length)


def batch_generator(data, batch_size, time_step):
    # code warning: never batched for the data tail
    batch_count = 0
    while True:
        if batch_count + batch_size + time_step >= data[0].shape[0] + 1:
            batch_count = 0
        start = np.arange(batch_count, batch_count + batch_size)
        end = np.arange(batch_count + time_step, batch_count + batch_size + time_step)
        batch_count += batch_size * time_step
        dx = data[0]
        dy = data[1]
        batch_x = np.array([dx[start[i]: end[i]] for i in range(start.shape[0])])
        batch_x = batch_x[:, :, np.newaxis]
        batch_y = np.array([dy[start[i]: end[i]] for i in range(start.shape[0])])
        batch_y = batch_y[:, :, np.newaxis]
        yield [batch_x, batch_y]

# batch_size = 1
# n_steps = 5
# batches = batch_generator([data_x, data_y], batch_size, n_steps)
# xb, yb = batches.next()
# print xb
# print xb[0,0,0] == 1
# exit()


def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

lr = 1e-3
batch_size = 1
n_steps = 50
input_size = 1
output_size = 1
cell_size = 100
total_epoch = 5000

x = tf.placeholder(tf.float32, [None, n_steps, input_size])
y = tf.placeholder(tf.float32, [None, n_steps, output_size])

x_2D = tf.reshape(x, [-1, input_size])
w_in = tf.Variable(tf.random_normal([input_size, cell_size]))
b_in = tf.Variable(tf.zeros([cell_size]))
h_2D = tf.matmul(x_2D, w_in) + b_in
h = tf.reshape(h_2D, [-1, n_steps, cell_size])

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(cell_size)
cell_init_state = lstm_cell.zero_state(batch_size, tf.float32)
cell_outputs, cell_final_state = tf.nn.dynamic_rnn(lstm_cell, h, initial_state=cell_init_state)

out_2D = tf.reshape(cell_outputs, [-1, cell_size])
w_out = tf.Variable(tf.random_normal([cell_size, output_size]))
b_out = tf.Variable(tf.zeros([output_size]))
pred = tf.matmul(out_2D, w_out) + b_out
pred_reshape = tf.reshape(pred, [-1])
y_reshape = tf.reshape(y, [-1])

x_test = tf.placeholder(tf.float32, [None, input_size])
h_2D_test = tf.matmul(x_test, w_in) + b_in
h_test = tf.reshape(h_2D_test, [-1, n_steps, cell_size])
cell_init_state_test = lstm_cell.zero_state(10, tf.float32)
cell_output_test, cell_final_state_test = tf.nn.dynamic_rnn(lstm_cell, h_test, initial_state=cell_init_state_test)
cell_output_test_2D = tf.reshape(cell_output_test, [-1, cell_size])
pred_test = tf.matmul(cell_output_test_2D, w_out) + b_out
pred_reshape_test = tf.reshape(pred_test, [-1])

# loss = tf.reduce_mean(tf.squared_difference(pred_reshape, y_reshape))

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([tf.reshape(pred, [-1])], [tf.reshape(y, [-1])],
            [tf.ones([batch_size * n_steps], dtype=tf.float32)], average_across_timesteps=True,
            softmax_loss_function=ms_error)
final_loss = tf.div(tf.reduce_sum(loss), batch_size)
train_op = tf.train.AdamOptimizer(lr).minimize(final_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batches = batch_generator([data_x, data_y], batch_size, n_steps)

    for epoch in range(total_epoch):
        xb, yb = batches.next()

        if epoch == 0 or xb[0, 0, 0] == 1:
            feed_dict = {x: xb, y: yb}

        else:
            feed_dict = {x: xb, y: yb, cell_init_state: state}

        _, cost, state = sess.run([train_op, final_loss, cell_final_state], feed_dict=feed_dict)

        if epoch % 20 == 0:
            print epoch, cost

    total_x = data_x[:, np.newaxis]
    final_pred = sess.run(pred_reshape_test, feed_dict={x_test: total_x})

plt.plot(data_x, data_y)
plt.plot(data_x, final_pred)
plt.show()
