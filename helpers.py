from __future__ import division
import pandas as pd
import numpy as np

def one_hot(idx, depth):
    '''
    Generates one hot vector
    '''
    vect = np.zeros(depth)
    vect[idx] = 1
    return vect

def load_data(file_name, shape):
    '''
    Loads images and targets from csv and normalizes images.
    params:
      file_name: path to file
      shape: image size
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
    params:
      y: the ground truth targets for a batch
      y_c: the unnormalized label frequencies for the batch
      mode: a string, either None or "disturb" or "soft"
      alpha: noise rate
    '''
    # Normalize crowd labels
    y_n = y_c / np.sum(y_c, axis=1, keepdims=True)
    classes = y.shape[1]
    corr_labels = np.argmax(y,axis=1)
    if mode == 'disturb':
        for i in range(len(y_n)):
            y_n[i] *= alpha
            y_n[i][corr_labels[i]] += 1 - alpha
            new_targ_idx = int(np.random.choice(a=classes, p=y_n[i]))
            y_n[i] = one_hot(new_targ_idx,classes)
    elif mode == 'soft':
        y_n = (y + alpha * y_n)/(1 + alpha)
    elif mode == 'disturb_uniform':
        for i in range(len(y)):
            y_n[i] += alpha/classes
            y_n[i][corr_labels[i]] += 1 - alpha
            new_targ_idx = int(np.random.choice(a=classes, p=y))
            y_n[i] = one_hot(new_targ_idx,classes)
    return y_n 
