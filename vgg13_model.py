import tensorflow as tf
import numpy as np
def conv2d(x, W):
    '''Generates convolutional layer according to vgg.'''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name=None):
    '''Generates max pooling layer with fixed 2x2 kernel'''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)

def weight_variable(shape):
    '''Generates a weight variable of a given shape.'''
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial, name='weights')

def bias_variable(shape):
    '''Generates a bias variable of a given shape.'''
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name='biases')

def vgg(X):
    '''
      Custom with dropout and less units in FC layers.
      X: float32 tensor containing the training examples
      Returns:
      y_out: output layer
      keep_prob: tensorflow float, default 1.0 means no drop
    '''
    # conv1_1

    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable(shape=[3, 3, 1, 32])
        conv = conv2d(X, kernel)
        biases = bias_variable([32])
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 32, 32])
        conv = conv2d(conv1_1, kernel)
        biases = bias_variable([32])
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)

    # pool1
    pool1 = max_pool_2x2(conv1_2, name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 32, 64])
        conv = conv2d(pool1, kernel)
        biases = weight_variable([64])
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)

        # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 64, 64])
        conv = conv2d(conv2_1, kernel)
        biases = bias_variable([64])
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope)

        # pool2
    pool2 = max_pool_2x2(conv2_2, name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = weight_variable([3, 3, 64, 128])
        conv = conv2d(pool2, kernel)
        biases = bias_variable([128])
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)

        # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = weight_variable([3, 3, 128, 128])
        conv = conv2d(conv3_1, kernel)
        biases = bias_variable([128])
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)

        # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = weight_variable([3, 3, 128, 128])
        conv = conv2d(conv3_2, kernel)
        biases = bias_variable([128])
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)

        # pool3
    pool3 = max_pool_2x2(conv3_3, name='pool3')

        # fc1
    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool3.get_shape()[1:]))
        fc1w = weight_variable([shape, 512])
        fc1b = bias_variable([512])
        pool3_flat = tf.reshape(pool3, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)

        # dropout layer after first fully connected
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        # fc2
    with tf.name_scope('fc2') as scope:
        fc2w = weight_variable([512, 512])
        fc2b = bias_variable([512])
        fc2l = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)

        # fc3
    with tf.name_scope('fc3') as scope:
        fc3w = weight_variable([512, 6])
        fc3b = bias_variable([6])
        y_out = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
    return y_out, keep_prob
