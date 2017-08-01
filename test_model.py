import tensorflow as tf
import math
import argparse
import os
import sys
from helpers import *
from vgg13_model import vgg

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = None
data_file_name = 'ck_data_48_48.csv'
test_file_name = 'jaffe_48_48.csv'
crowd_file_name = 'crowd.csv'
output_dir = 'tmp/models'
expression_table = {'Anger'    : 0,
                    'Disgust'  : 1,
                    'Fear'     : 2,
                    'Happiness': 3,
                    'Sadness'  : 4,
                    'Surprise' : 5}

noise = 0.11
model_path = ''     # where the model is saved
model_name = ''     # ckpt file name
SHAPE = (48, 48)

def main(_):
    train_mode = FLAGS.train_mode
    print FLAGS.data_dir

    # Import data and labels
    X_train, y_train = load_data(os.path.join(FLAGS.data_dir, data_file_name),SHAPE)
    X_test, y_test = load_data(os.path.join(FLAGS.data_dir, test_file_name),SHAPE)
    train_data_size, test_data_size = X_train.shape[0], X_test.shape[0]

    # Crowdsource part
    is_crowd_train = False
    if train_mode == 'disturb' or train_mode == 'soft':
        print 'crowd training enabled'
        is_crowd_train = True
        y_temp = load_crowd_labels(os.path.join(FLAGS.data_dir, crowd_file_name))
        y_train = process_target(y_train, y_temp, alpha=noise, mode=train_mode)

    print 'train data dims:', X_train.shape, 'train output dims:', y_train.shape
    print 'test data dims:', X_test.shape, 'test output dims:', y_test.shape

    # create model
    X = tf.placeholder(tf.float32, [None, X_train.shape[1], X_train.shape[2], 1])
    y = tf.placeholder(tf.int64, [None, 6])

    # get output
    y_out, keep_prob = vgg(X)

    # loss variable
    mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                labels=y,
                                logits=y_out))

    # compute accuracy
    correct_prediction = tf.equal(tf.argmax(y_out, axis=1),tf.argmax(y,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # confusion matrix
    confusion_matrix = tf.confusion_matrix(labels=tf.argmax(y,axis=1),
                                        predictions=tf.argmax(y_out,axis=1),
                                        num_classes=6)
    # restore model
    saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph(os.path.join(model_path, meta_file))

    losses = {'train':0, 'test':0}

    with tf.Session() as sess:
        with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"\
            saver.restore(sess,model_path+model_name)

            # compute the losses
            train_loss, train_acc = sess.run([mean_loss, accuracy],
                        feed_dict={X:X_train, y:y_train, keep_prob:1.0})
            test_loss, test_acc = sess.run([mean_loss, accuracy],
                        feed_dict={X:X_test, y:y_test, keep_prob:1.0})

            losses['train'] = (train_acc)
            losses['test'] = (test_acc)
            print losses

            confusion_results = sess.run(confusion_matrix,
                                feed_dict={X:X_test, y:y_test, keep_prob:1.0})
            np.savetxt(fname=os.path.join(model_path,'test_confusion'),
                       X=confusion_results,
                       fmt='%d',
                       delimiter=',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='tmp/data/',
                      help='Directory for storing all training data')
    parser.add_argument('--train_mode', type=str,
                          default='none',
                          help='\'none\', \'disturb\', or \'soft\'')
    FLAGS = parser.parse_args()
    tf.app.run(main=main)
