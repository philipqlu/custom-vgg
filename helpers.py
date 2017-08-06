from __future__ import division
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

def show_image(img):
    plt.figure()
    plt.imshow(img.reshape((img.shape[0],img.shape[1])), cmap='gray')
    plt.show()

def image_augment(image):
    '''
    Performs some transformations on the image
    '''
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image

def augment_data(X):
    '''
    Does data augmentation on input images X.
    '''
    X_out = tf.map_fn(image_augment, X)
    crop_boxes = np.zeros((X.shape[0], 4))
    for box in crop_boxes:
        # [y1, x1, y2, x2]
        box[0] = random.uniform(0,0.12,)
        box[1] = random.uniform(0,0.12)
        box[2] = random.uniform(0.88,1)
        box[3] = random.uniform(0.88,1)
    X_out = tf.image.crop_and_resize(X_out,
                                     boxes=crop_boxes,
                                     box_ind=np.arange(X.shape[0]),
                                     crop_size=[X.shape[1], X.shape[2]])
    return X_out

def preprocess_images(X_train, X_test, augment=False, augment_type='concat'):
    '''
    Does standard normalization on train and test set.
    '''
    img_mean, img_std = np.mean(X_train, axis=0), np.std(X_train, axis=0)
    if augment:
        X_n = augment_data(X_train).eval()
        if augment_type == 'concat':
            X_train = np.concatenate((X_train,X_n), axis=0)
            y_train = np.concatenate((y_train,y_train), axis=0)
        else:
            X_train = X_n
    X_train = (X_train - img_mean) / img_std
    X_test = (X_test  - img_mean) / img_std
    return X_train, X_test

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
            y_n[i][corr_labels[i]] -= alpha
            new_targ_idx = int(np.random.choice(a=classes, p=y_n[i]))
            y_n[i] = one_hot(new_targ_idx,classes)
    return y_n
