import tensorflow as tf
import numpy as np
import sys

def build_model(num_classes, model_name):
    '''
    Factory function to instantiate the model.
    '''
    model = getattr(sys.modules[__name__], model_name)
    return model(num_classes)

class CustomVGG13(object):
    @property
    def input_width(self):
        return 48

    @property
    def input_height(self):
        return 48

    @property
    def input_channels(self):
        return 1

    @property
    def input(self):
        return self._input

    @property
    def target(self):
        return self._target

    @property
    def logits(self):
        return self._logits

    @property
    def keep_prob(self):
        return self._keep_prob

    def __init__(self, num_classes):
        self._input, self._target, self._logits, self._keep_prob = self._build(num_classes)

    def _conv2d(self, x, W):
        '''Generates convolutional layer according to vgg.'''
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _max_pool_2x2(self, x, name=None):
        '''Generates max pooling layer with fixed 2x2 kernel'''
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _weight_variable(self, shape):
        '''Generates a weight variable of a given shape.'''
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name='weights')

    def _bias_variable(self, shape):
        '''Generates a bias variable of a given shape.'''
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial, name='biases')

    def _build(self, num_classes):
        '''
        Architecture: 2 + 2 + 3 conv + 3 max pool + 2 FC + 1 FC + out
        Dropout after pooling and FC
          X: float32 tensor containing the training examples
          y_out: output layer
          keep_prob: tensorflow float that we will feed back later
        '''
        X = tf.placeholder(tf.float32, [None, self.input_height, self.input_width, self.input_channels],name='X')
        y = tf.placeholder(tf.int64, [None, num_classes],name='y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # conv1_1

        with tf.name_scope('conv1_1') as scope:
            kernel = self._weight_variable(shape=[3, 3, 1, 32])
            conv = self._conv2d(X, kernel)
            biases = self._bias_variable([32])
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out, name=scope)

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = self._weight_variable([3, 3, 32, 32])
            conv = self._conv2d(conv1_1, kernel)
            biases = self._bias_variable([32])
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out, name=scope)

        # pool1
        pool1 = self._max_pool_2x2(conv1_2, name='pool1')
        # pool1 = tf.nn.dropout(pool1, keep_prob)

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = self._weight_variable([3, 3, 32, 64])
            conv = self._conv2d(pool1, kernel)
            biases = self._weight_variable([64])
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out, name=scope)

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = self._weight_variable([3, 3, 64, 64])
            conv = self._conv2d(conv2_1, kernel)
            biases = self._bias_variable([64])
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out, name=scope)

        # pool2
        pool2 = self._max_pool_2x2(conv2_2, name='pool2')
        # pool2 = tf.nn.dropout(pool2, keep_prob)

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = self._weight_variable([3, 3, 64, 128])
            conv = self._conv2d(pool2, kernel)
            biases = self._bias_variable([128])
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out, name=scope)

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = self._weight_variable([3, 3, 128, 128])
            conv = self._conv2d(conv3_1, kernel)
            biases = self._bias_variable([128])
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out, name=scope)

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = self._weight_variable([3, 3, 128, 128])
            conv = self._conv2d(conv3_2, kernel)
            biases = self._bias_variable([128])
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out, name=scope)

        # pool3
        pool3 = self._max_pool_2x2(conv3_3, name='pool3')
        # pool3 = tf.nn.dropout(pool3, keep_prob)

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = self._weight_variable([3, 3, 128, 128])
            conv = self._conv2d(pool3, kernel)
            biases = self._bias_variable([128])
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out, name=scope)

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = self._weight_variable([3, 3, 128, 128])
            conv = self._conv2d(conv4_1, kernel)
            biases = self._bias_variable([128])
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out, name=scope)

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = self._weight_variable([3, 3, 128, 256])
            conv = self._conv2d(conv4_2, kernel)
            biases = self._bias_variable([256])
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out, name=scope)

        # pool3
        pool4 = self._max_pool_2x2(conv4_3, name='pool3')
        # pool4 = tf.nn.dropout(pool4, keep_prob)

        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(pool4.get_shape()[1:]))
            fc1w = self._weight_variable([shape, 1024])
            fc1b = self._bias_variable([1024])
            pool4_flat = tf.reshape(pool4, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool4_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1l)

        # dropout_2
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = self._weight_variable([1024, 1024])
            fc2b = self._bias_variable([1024])
            fc2l = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
            fc2 = tf.nn.relu(fc2l)

        # dropout_2
        fc2_drop = tf.nn.dropout(fc2, keep_prob)

            # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = self._weight_variable([1024, num_classes])
            fc3b = self._bias_variable([num_classes])
            y_out = tf.nn.bias_add(tf.matmul(fc2_drop, fc3w), fc3b, name='y_out')
        return X, y, y_out, keep_prob
