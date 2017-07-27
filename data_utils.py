import pandas as pd
import numpy as np
from PIL import Image

class DataReader(object):
    def __init__(data_dir, csv_file = 'train.csv', crowd_file = 'crowd.csv'):
        self.file_path = data_dir
        self.csv_file = csv_file
        self.crowd_file = crowd_file
        self.images = None
        self.targets = None
        self.crowd_targets = None

    def download_images(self):
        pass
    def preprocess_images(self):
        pass
    def load_data(self):
        pass


def load_data(file_name, shape=(128,98)):
    '''
    Loads image data from csv and returns Xd and yd 4-d arrays
    '''
    df = pd.read_csv(file_name)
    num_examples, X_shape, X_depth = len(df), shape, 1
    Xd = np.empty((num_examples, X_shape[0], X_shape[1], X_depth))
    yd = np.empty((num_examples))
    for i in range(num_examples):
        str_list = df.pixels[i].split(' ')
        pixel_flat = np.array([int(x) for x in str_list])
        pixel_2d = np.reshape(pixel_flat, newshape=(X_shape[0], X_shape[1], 1))
        #vect = [0 for i in range(7)]
        #vect[int(df.emotion[i])] = 1  # emotion label for example i
        yd[i]= int(df.emotion[i])
        Xd[i] = pixel_2d
    # Normalize the data
    Xd =  (Xd - np.mean(Xd, keepdims=True)) / np.std(Xd, keepdims=True)
    return Xd, yd

def load_crowd_labels(file_name):
    '''
    Loads crowd targets from csv and returns 2D np array
    '''
    df = pd.read_csv(file_name)
    num_examples, num_classes = len(df), len(df.iloc[0,1:])
    print num_examples, num_classes
    yc = np.empty((30, 6), dtype=float)
    for i in range(30):
        yc[i] = np.array(df.iloc[i,1:])
    print "crowd labels loaded"
    return yc
