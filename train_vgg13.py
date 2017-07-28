from __future__ import division
import tensorflow as tf
import math
import time
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import KFold
import pandas as pd
import os
import sys
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None
data_file_name = 'ck_data_128_98.csv'
output_dir = 'tmp/models'
expression_table = {'Anger'    : 0,
                    'Disgust'  : 1,
                    'Fear'     : 2,
                    'Happiness': 3,
                    'Sadness'  : 4,
                    'Surprise' : 5}

NOISES = [0, 0.05, 0.1, 0.2, 0.5, 0.8]

def preprocess_stats(X, length, normalize=True):
    '''
    Returns the mean and stdev of a target X in the desired length.
    '''
    if normalize:
        newX = (X - np.mean(X, axis=(1,2,3), keepdims=True)) / np.std(X, axis=(1,2,3), keepdims=True)
    else:
        newX = X - np.mean(X, axis=(1,2,3), keepdims=True)
    return newX

def one_hot(idx, depth):
    '''
    Generates one hot vector
    '''
    vect = np.zeros(depth)
    vect[idx] = 1
    return vect

def load_data(file_name, shape=(int(128),int(98))):
    '''
    Loads images and targets from csv and normalizes images.
    '''
    df = pd.read_csv(file_name)
    num_examples, num_classes, X_depth = len(df), len(set(df.emotion)), 1
    Xd = np.empty((num_examples, shape[0], shape[1], X_depth))
    yd = np.empty((num_examples, num_classes))
    for i in range(num_examples):
        str_list = df.pixels[i].split(' ')
        pixel_flat = np.array([int(x) for x in str_list])
        pixel_2d = np.reshape(pixel_flat, newshape=(shape[0], shape[1], 1))
        Xd[i] = pixel_2d
        yd[i] = one_hot(int(df.emotion[i]), num_classes)
    return Xd, yd


def load_crowd_labels(file_name):
    '''
    Loads crowd targets from csv and returns 2D np-array
    '''
    df = pd.read_csv(file_name)
    num_examples, num_classes = len(df), len(df.iloc[0,1:])
    print num_examples, num_classes
    yc = np.empty((num_examples, 6), dtype=float)
    for i in range(num_examples):
        yc[i] = np.array(df.iloc[i,1:])
    print "crowd labels loaded. example:", list(yc[0])
    return yc

def process_target(y, y_c, alpha=0.1, mode='disturb'):
    '''
      y: the ground truth targets for a batch
      y_c: the unnormalized label frequencies for the batch
      mode: a string, either None or "disturb" or "soft"
      alpha: noise rate
    '''
    # Normalize it
    y_n = y_c / np.sum(y_c, axis=1, keepdims=True)
    classes = y.shape[1]
    if mode == 'disturb':
        for i in range(len(y_n)):
            new_targ_idx = int(np.random.choice(a=np.ones(classes), p=y_n[i]))
            #print new_targ_idx
            y_n[i] = one_hot(new_targ_idx,classes)
    elif mode == 'soft':
        y_n = (y + alpha * y_n)/(1 + alpha)
    return y_n

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

def main(_):
    print FLAGS.data_dir
    train_mode = FLAGS.train_mode
    # Import data
    Xd, yd = load_data(os.path.join(FLAGS.data_dir, data_file_name))
    data_size = Xd.shape[0]

    # Crowdsource part
    is_crowd_train = False
    if train_mode == 'disturb' or train_mode == 'soft':
        print 'crowd training enabled'
        is_crowd_train = True
        yc = load_crowd_labels(os.path.join(FLAGS.data_dir, 'crowd.csv'))

    print 'input data dims:', Xd.shape, 'output data dims:', yd.shape
    print '===\n' * 3

    # Create model
    X = tf.placeholder(tf.float32, [None, Xd.shape[1], Xd.shape[2], 1])
    y = tf.placeholder(tf.int64, [None, 6])

    # Get output
    y_out, keep_prob = vgg(X)

    # loss variable
    mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                labels=y,
                                logits=y_out))

    # compute accuracy
    correct_prediction = tf.equal(tf.argmax(y_out, axis=1),tf.argmax(y,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # required dependencies for batch normalization
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = tf.train.AdamOptimizer(4e-4).minimize(mean_loss)

    # run parameters
    epochs = 30
    batch_size = 64
    noise_rate = 0.1

    # shuffle indices
    data_indices = np.arange(data_size)
    np.random.shuffle(data_indices)
    Xd = Xd[data_indices]
    yd = yd[data_indices]
    if is_crowd_train:
        yc = yc[data_indices]


    # # training and testing sets for fixed split
    # slice_idx = int(math.ceil(data_size*0.7))
    # X_train = Xd[0:slice_idx]
    # y_train = yd[0:slice_idx]
    # X_test = Xd[slice_idx:]
    # y_test = yd[slice_idx:]
    # if is_crowd_train:
    #     yc = yc[data_indices]
    #     yc = yc[:slice_idx]
    #     if train_mode=='soft':
    #         y_train = process_target(y_train, y_c, noise_rate, 'soft')
    #
    # # preprocess
    # X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=(0))
    # X_test  = (X_test  - np.mean(X_train, axis=0)) / np.std(X_train, axis=(0))

    # save model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"
            splits = len(Xd) - 9
            print 'splits:', len(Xd) - 20
            k_fold = KFold(n_splits=splits)
            fold = 0
            total_val_acc = 0
            for train_indices, test_indices in k_fold.split(Xd):
                # We're retraining our model each time now. It's comp expensive
                # but only way with such a small dataset.
                sess.run(tf.global_variables_initializer())
                fold += 1

                # training and val splits
                X_train = Xd[train_indices]
                y_train = yd[train_indices]
                X_test = Xd[test_indices]
                y_test = yd[test_indices]

                if is_crowd_train:
                    yc_train = yc[train_indices]
                    if train_mode=='soft':
                        y_train = process_target(y_train, y_c, noise_rate, 'soft')

                # preprocess
                X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=(0))
                X_test  = (X_test  - np.mean(X_train, axis=0)) / np.std(X_train, axis=(0))

                print('Training')

                # track some stats
                iter_cnt = 0
                losses = {'train':[],'test':[]}
                best_epoch = 0
                max_val_acc = 0

                # start timing
                start_time = time.time()

                train_indices = np.arange(len(X_train))
                for e in range(epochs):
                    for i in range(int(math.ceil(X_train.shape[0] / batch_size))):
                        start_idx = (i * batch_size) % X_train.shape[0]
                        indices = train_indices[start_idx:start_idx + batch_size]
                        # current mini batch
                        X_mini, y_mini = X_train[indices, :], y_train[indices]
                        actual_batch_size = y_mini.shape[0]

                        # process new targets
                        if is_crowd_train:
                            yc_mini = yc[indices]
                            if train_mode=='disturb':
                                y_mini = process_target(y_mini, yc_mini, noise_rate, train_mode)

                        train_step.run(feed_dict={X: X_mini, y: y_mini, keep_prob:0.8})
                        iter_cnt += 1

                    # compute the losses
                    train_loss, train_acc = sess.run([mean_loss, accuracy],
                                feed_dict={X:X_mini, y:y_mini, keep_prob:1.0})
                    test_loss, test_acc = sess.run([mean_loss, accuracy],
                                feed_dict={X:X_test, y:y_test, keep_prob:1.0})

                    losses['train'].append(1-train_acc)
                    losses['test'].append(1-test_acc)
                    if max_val_acc < test_acc:
                        max_val_acc = test_acc
                        best_epoch = e
                    print("Fold {5} Epoch {0}, Train loss = {1:.5g}, Train acc = {2:.5f}, Test loss = {3: .5g}, Test Acc = {4:.5f}" \
                          .format(e, train_loss, train_acc, test_loss, test_acc, fold))
                print "Fold {0} Summary: Best Epoch: {1} with Error {2:.5g}".format(fold, best_epoch, max_val_acc)
                total_val_acc += max_val_acc
                end_time = time.time()
            print 'Cross-val error with {0} folds = {1:.3f}'.format(fold,total_val_acc / (fold))
            print 'Train time: {:.3f}'.format(end_time-start_time)

            save_path = saver.save(sess, os.path.join(output_dir,FLAGS.model_name))
            print("Model saved in file: %s" % save_path)
            # print('Test')

            plt.figure(1)
            plt.grid(True)
            plt.title('Loss')
            plt.xlabel('Epoch number')
            plt.ylabel('Recognition Error Rate')
            for key, value in losses.items():
                plt.plot(value, label=key)
            plt.legend()
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='tmp/data/',
                      help='Directory for storing all training data')
    parser.add_argument('--model_name', type=str,
                          default='model',
                          help='What to save as the model name')
    parser.add_argument('--train_mode', type=str,
                          default='none',
                          help='\'none\', \'disturb\', or \'soft\'')
    FLAGS = parser.parse_args()
    tf.app.run(main=main)
