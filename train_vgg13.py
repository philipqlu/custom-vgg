from __future__ import division
import tensorflow as tf
import math
import time
import matplotlib.pyplot as plt
import argparse
# from sklearn.model_selection import KFold
import pandas as pd
import os
import sys
import numpy as np

FLAGS = None

# 0=neutral, 1=anger, 2=disgust, 3=fear, 4=happy, 5=sadness, 6=surprise
labels = ['Happiness', 'Fear', 'Surprise', 'Disgust', 'Anger', 'Sadness']
label_dict = {1:'Anger', 2:'Disgust', 3:'Fear', 4:'Happiness', 5:'Sadness', 6:'Surprise'}

def preprocess_images(X, normalize=False):
    '''
    Simple normalization or mean subtraction. Note only do this on training set.
    '''
    if normalize:
        return (X - np.mean(X, axis=(1,2,3), keepdims=True)) / np.std(X, axis=(1,2,3), keepdims=True)
    else:
        return X - np.mean(X, axis=(1,2,3), keepdims=True)


def load_data(file_name, shape=(int(128/4),int(98/4))):
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
        # pixel_flat = pixel_flat[:400]
        pixel_2d = np.reshape(pixel_flat, newshape=(shape[0], shape[1], 1))
        yd[i]= int(df.emotion[i])

        Xd[i] = pixel_2d
    return Xd, yd


def load_crowd_labels(file_name):
    '''
    Loads crowd targets from csv and returns 2D np-array
    '''
    df = pd.read_csv(file_name)
    num_examples, num_classes = len(df), len(df.iloc[0,1:])
    print num_examples, num_classes
    yc = np.empty((40, 6), dtype=float)
    for i in range(40):
        yc[i] = np.array(df.iloc[i,1:])
    print "crowd labels loaded"
    return yc

def process_target(y, y_c, alpha=0.1, mode='disturb'):
    '''
      y: the ground truth targets for a minibatch
      y_c: the unnormalized label frequency vector for the minibatch
      mode: a string, either None or "disturb" or "soft"
      alpha: noise rate
    '''
    # Normalize it
    y_n = y_c / np.sum(y_c, axis=0)
    if mode == 'disturb':
        new_targ_idx = np.random.choice(a=np.ones(len(y)), p=y_n)
        return tf.one_hot(new_targ_idx,7)
    elif mode == 'soft':
        y_new = (y + alpha * y_n)/(1 + alpha)
        return y_new

def conv2d(x, W):
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
        print 'params:', shape, 'shape:', pool3.get_shape()
        fc1w = weight_variable([shape, 1024])
        fc1b = bias_variable([1024])
        pool3_flat = tf.reshape(pool3, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)

        # dropout layer after first fully connected
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        fc1_drop = tf.nn.dropout(fc1, keep_prob)

        # fc2
    with tf.name_scope('fc2') as scope:
        fc2w = weight_variable([1024, 1024])
        fc2b = bias_variable([1024])
        fc2l = tf.nn.bias_add(tf.matmul(fc1_drop, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)

        # fc3
    with tf.name_scope('fc3') as scope:
        fc3w = weight_variable([1024, 6])
        fc3b = bias_variable([6])
        y_out = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
    return y_out, keep_prob

def plot_losses():
    plt.subplot(3, 1, 1)
    plt.plot(losses)
    plt.grid(True)
    plt.title('Loss')
    plt.xlabel('Epoch number')
    plt.ylabel('cross-entropy loss')
    plt.show()

def main(_):
    print FLAGS.data_dir
    train_mode = FLAGS.train_mode
    # Import data
    Xd, yd = load_data(os.path.join(FLAGS.data_dir, 'ck_tiny_data.csv'))
    print 'input data dims:', Xd.shape, 'output data dims:', yd.shape

    # Create Model
    X = tf.placeholder(tf.float32, [None, Xd.shape[1], Xd.shape[2], 1])
    y = tf.placeholder(tf.int64, [None, 6])

    # Get output and loss
    y_out, keep_prob = vgg(X)
    mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                labels=y,
                                logits=y_out))

    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(y_out, axis=1),tf.argmax(y,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # required dependencies for batch normalization
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = tf.train.AdamOptimizer(4e-4).minimize(mean_loss)

    # save model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"
            sess.run(tf.global_variables_initializer())
            # print("TRAIN:", train_indices, "TEST:", test_indices)
            print('Training')
            # shuffle indicies
            train_indicies = np.arange(Xd.shape[0])
            np.random.shuffle(train_indicies)

            # Crowdsource part
            is_crowd_train = False
            if train_mode == 'disturb' or train_mode == 'soft':
                print 'crowd training enabled'
                is_crowd_train = True
                yc = load_crowd_labels(os.path.join(FLAGS.data_dir, 'crowd.csv'))

            # counter
            iter_cnt = 0
            losses = []
            # start timing
            start_time = time.time()
            epochs = 50
            batch_size = 48
            for e in range(epochs):
                # make sure we iterate over the dataset once
                for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                    start_idx = (i * batch_size) % Xd.shape[0]
                    indices = train_indicies[start_idx:start_idx + batch_size]

                    # current mini batch
                    X_mini, y_mini = Xd[indices, :], yd[indices]
                    actual_batch_size = y_mini.shape[0]

                    # process new targets
                    if is_crowd_train:
                        yc_mini = yc[indices]
                        y_mini = [process_target(yc_mini, alpha, train_mode) for y in y_mini]

                    train_step.run(feed_dict={X: X_mini, y: y_mini, keep_prob:1.0})
                    iter_cnt += 1

                loss, acc = sess.run([mean_loss, accuracy],
                            feed_dict={X:X_mini, y:y_mini, keep_prob:1.0})
                losses.append(loss)
                print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
                      .format(loss, acc, e + 1))

            plot_losses(losses)
            end_time = time.time()
            print 'train time: {:.3f}'.format(start_time-end_time)
            return total_loss, total_correct

            save_path = saver.save(sess, os.path.join(FLAGS.data_dir,FLAGS.model_name))
            print("Model saved in file: %s" % save_path)
            print('Validation')
            # run_model(sess, y_out, mean_loss, Xd[test_indices], yd[test_indices], epochs = 1, batch_size
            #print('test')
            # run_model(sess, y_out,mean_loss,Xd[0:12,:],yd[0:12],1,12) # test

# variables
# epochs, is_training, is_crowd_train, alpha, train_mode, batch_size

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
