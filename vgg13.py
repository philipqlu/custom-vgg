import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

import pandas as pd

X = tf.placeholder(tf.float32, [None, 490, 640, 1])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)


def vgg(X):
    # input images
    # output 7 classes of scores

    # conv1_1

    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal(shape = [3, 3, 1, 32], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(X, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)

    # conv1_2
    with tf.name_scope('conv1_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)

    # pool1
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # conv2_1
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_1 = tf.nn.relu(out, name=scope)

        # conv2_2
    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv2_2 = tf.nn.relu(out, name=scope)

        # pool2
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3_1
    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)

        # conv3_2
    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)

        # conv3_3
    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                 stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)

        # pool3
    pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')



    with tf.name_scope('fc1') as scope:
        shape = int(np.prod(pool3.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='biases')
        pool3_flat = tf.reshape(pool3, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)

        # fc2
    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                           trainable=True, name='biases')
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)

        # fc3
    with tf.name_scope('fc3') as scope:
        fc3w = tf.Variable(tf.truncated_normal([4096, 7],
                                               dtype=tf.float32,
                                               stddev=1e-1), name='weights')
        fc3b = tf.Variable(tf.constant(1.0, shape=[7], dtype=tf.float32),
                           trainable=True, name='biases')
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)

    y_out = fc3l
    return y_out

y_out = vgg(X)

def load_data(file_name):
    '''Loads image data from csv and returns Xd and yd 4-d arrays
    '''
    df = pd.read_csv(file_name)
    num_examples, X_shape, X_depth = len(df), (490,640), 1
    Xd = np.empty((num_examples, X_shape[0], X_shape[1], X_depth))
    yd = np.empty((num_examples))
    for i in range(num_examples):
        str_list = df.pixels[i].split(' ')[0:-1]
        pixel_flat = np.array([int(x) for x in str_list])
        pixel_2d = np.reshape(pixel_flat, newshape=(490, 640, 1))
        #vect = [0 for i in range(7)]
        #vect[int(df.emotion[i])] = 1  # emotion label for example i
        yd[i]= int(df.emotion[i])
        Xd[i] = pixel_2d
    # Normalize the data
    Xd -=  np.mean(Xd, axis=0)
    return Xd, yd

Xd, yd = load_data('ck_image_data_sample.csv')

# total_loss = tf.losses.hinge_loss(tf.one_hot(y,7),logits=y_out)
# mean_loss = tf.reduce_mean(total_loss)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y,7) , logits=y_out))
mean_loss = cross_entropy
# required dependencies for batch normalization
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = tf.train.AdamOptimizer(5e-4).minimize(mean_loss)

def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict, 1),y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val, correct_prediction, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
            # generate indicies for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indicies[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx, :],
                         y: yd[idx],
                         is_training: training_now}
            # get batch size
            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            correct += np.sum(corr)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                      .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))
            iter_cnt += 1
        total_correct = float(correct / Xd.shape[0])
        total_loss = float(np.sum(losses) / Xd.shape[0])
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
              .format(total_loss, total_correct, e + 1))
        if plot_losses:
            plt.subplot(3, 1, 1)
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')

            plt.show(block = False) #you can delete the thing in bracket
    return total_loss, total_correct


with tf.Session() as sess:
    with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,
                  predict=y_out,
                  loss_val = mean_loss,
                  Xd = Xd,
                  yd = yd,
                  epochs = 50,
                  batch_size=24,
                  print_every=10,
                  training=train_step,
                  plot_losses=False)
         #print('validation')
        #run_model(sess, y_out,mean_loss,Xd[0:12,:],yd[0:12],1,12) # test
        #print('test')
        # run_model(sess, y_out,mean_loss,Xd[0:12,:],yd[0:12],1,12) # test
