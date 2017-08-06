import tensorflow as tf
import math
import time
import argparse
from sklearn.model_selection import KFold
import os
from helpers import *
from vgg13_model import build_model

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
log_dir = 'tmp/logs'
augment = False
augment_type = 'replace'

def main(_):
    # Create model

    model = build_model(num_classes=6, model_name='Vgg13Small')
    X, y, y_out, keep_prob = model.input, model.target, model.logits, model.keep_prob
    image_shape = model.input_height, model.input_width

    # Import data and labels
    Xd, yd = load_data(os.path.join(FLAGS.data_dir, data_file_name),shape=image_shape)
    data_size = Xd.shape[0]

    # Crowdsource part
    is_crowd_train = False
    train_mode = FLAGS.train_mode
    if train_mode == 'disturb' or train_mode == 'soft':
        print 'crowd training enabled:', train_mode
        is_crowd_train = True
        yc = load_crowd_labels(os.path.join(FLAGS.data_dir, crowd_file_name))

    print 'input data dims:', Xd.shape, 'output data dims:', yd.shape



    # loss variable
    mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                labels=y,
                                logits=y_out), name='mean_loss')
    tf.summary.scalar('mean_loss1', mean_loss)

    # compute accuracy, top-1 to top-3
    correct_prediction = tf.equal(tf.argmax(y_out, axis=1),tf.argmax(y,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),
                              name='accuracy')
    tf.summary.scalar('accuracy1', accuracy)

    correct_prediction_2 = tf.nn.in_top_k(predictions=y_out,
                                          targets=tf.argmax(y,axis=1), k=2)
    accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, tf.float32))

    correct_prediction_3 = tf.nn.in_top_k(predictions=y_out,
                                          targets=tf.argmax(y,axis=1), k=3)
    accuracy_3 = tf.reduce_mean(tf.cast(correct_prediction_3, tf.float32))

    # confusion matrix
    confusion_matrix = tf.confusion_matrix(labels=tf.argmax(y,axis=1),
                                           predictions=tf.argmax(y_out,axis=1),
                                           num_classes=6,
                                           name='confusion_matrix')

    # required dependencies for batch normalization
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate=1e-4,
                                            epsilon=0.017).minimize(mean_loss)

    # run parameters
    epochs = 100
    batch_size = 32

    # shuffle indices
    data_indices = np.arange(data_size)
    np.random.shuffle(data_indices)
    Xd, yd = Xd[data_indices], yd[data_indices]
    if is_crowd_train:
        yc = yc[data_indices]

    # summary logs
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        with tf.device("/cpu:0"):  # "/cpu:0" or "/gpu:0"
            train_writer = tf.summary.FileWriter(log_dir + '/train',
                                      sess.graph)
            test_writer = tf.summary.FileWriter(log_dir + '/test',
                                      sess.graph)
            splits = 10
            print 'splits:', splits
            k_fold = KFold(n_splits=splits)
            fold = 0
            noises = [0.1, 0.2]
            for train_indices, test_indices in k_fold.split(Xd):
                fold += 1
                # training and validation splits
                X_train, y_train = Xd[train_indices], yd[train_indices]
                X_test, y_test = Xd[test_indices], yd[test_indices]
                X_train, X_test = preprocess_images(X_train, X_test)

                if is_crowd_train:
                    yc_train = yc[train_indices]

                batch_indices = np.arange(len(X_train))
                for noise_rate in noises:

                    sess.run(tf.global_variables_initializer())

                    print len(X_train), len(X_test)
                    print('Training')

                    losses = {'train':[],'test':[],'test2':[],'test3':[]}
                    best_epoch, max_val_acc = 0, 0

                    start_time = time.time()

                    for e in range(epochs):
                        for i in range(int(math.ceil(X_train.shape[0] / batch_size))):
                            start_idx = (i * batch_size) % X_train.shape[0]
                            indices = batch_indices[start_idx : start_idx + batch_size]
                            # current mini batch
                            X_mini, y_mini = X_train[indices, :], y_train[indices]
                            actual_batch_size = y_mini.shape[0]

                            # process new targets for the batch
                            if train_mode == 'disturb':
                                y_mini = process_target(y_mini, yc_train[indices], noise_rate, train_mode)
                            elif train_mode == 'disturb_uniform':
                                y_mini = process_target(y_mini, y_mini, noise_rate, train_mode)
                            elif train_mode == 'soft':
                                y_mini = process_target(y_mini, yc_train[indices], noise_rate, train_mode)

                            train_step.run(feed_dict={X: X_mini, y: y_mini, keep_prob:0.5})

                        # compute the losses
                        train_loss, train_acc, train_summary = sess.run([mean_loss, accuracy, merged],
                                    feed_dict={X:X_train, y:y_train, keep_prob:1.0})
                        test_loss, test_acc, test_acc2, test_acc3, test_summary = sess.run([mean_loss, accuracy, accuracy_2, accuracy_3, merged],
                                    feed_dict={X:X_test, y:y_test, keep_prob:1.0})

                        train_writer.add_summary(train_summary)
                        test_writer.add_summary(test_summary)

                        losses['train'].append(1-train_acc)
                        losses['test'].append(1-test_acc)
                        losses['test2'].append(1-test_acc2)
                        losses['test3'].append(1-test_acc3)
                        if max_val_acc < test_acc:
                            max_val_acc = test_acc
                            best_epoch = e
                        print "Fold {7} Epoch {0}, Tr loss = {1:.5g}, Tr acc = {2:.5f}, Te loss = {3: .5g}, Te Acc = {4:.5f}, Te-2 Acc = {5:.5f}, Te-3 Acc = {6:.5f}" \
                              .format(e, train_loss, train_acc, test_loss, test_acc, test_acc2, test_acc3, fold)
                    print "Fold {0} Summary: Best Epoch: {1} with Error {2:.5g}".format(fold, best_epoch, max_val_acc)

                    # Save our important summaries
                    model_name = FLAGS.model_name +'noise_' + str(noise_rate)
                    model_path = os.path.join(output_dir, model_name)
                    if not os.path.exists(model_path):
                        os.mkdir(model_path)

                    # confusion matrix
                    confusion_results = sess.run(confusion_matrix, feed_dict={X:X_test, y:y_test, keep_prob:1.0})
                    np.savetxt(os.path.join(model_path,'confusion'+str(fold)), confusion_results, fmt='%d', delimiter=',')
                    #save losses
                    out_df = pd.DataFrame({'train':losses['train'],'val':losses['test'],'val2':losses['test2'],'val3':losses['test3']})
                    out_df.to_csv(os.path.join(model_path, 'fold'+str(fold)),index=False)

                    # save model
                    saver = tf.train.Saver()
                    save_path = saver.save(sess, os.path.join(model_path,str(test_acc)+'fold'+str(fold)))
                    print("Model saved in: %s" % model_path)

                    end_time = time.time()
                    print 'Train time: {:.3f}'.format(end_time-start_time)

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
