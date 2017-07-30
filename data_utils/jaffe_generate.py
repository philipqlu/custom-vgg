from PIL import Image
import numpy as np
import pandas as pd
import argparse
import os

in_path = 'jaffeimages/jaffe'
out_file = 'jaffe_48_48.csv'

# Sample name format: KA.AN1.39
# General format:     XX.YY#.###
# Emotions: (YY)
# AN = Angry
# DI = Disgust
# FE = Fear
# HA = Happy
# SA = Sadness
# SU = Surprise
# Ignore: NE = Neutral

emot_dict = {'AN':0,'DI':1,'FE':2,'HA':3,'SA':4,'SU':5}

def get_emotion(string):
    emot = string[3:5]
    if emot not in emot_dict.keys():
        print emot, string
        return -1
    return emot_dict[emot]

def process(file_path):
    out_data = {'pixels':[],'emotion':[]}
    for img_file in os.listdir(file_path):
        # process the name
        emotion = get_emotion(img_file)
        if emotion != -1:
            img_path = os.path.join(file_path, img_file)
            img = Image.open(img_path).convert('L').resize((48,48),Image.ANTIALIAS)
            img_flat = np.asarray(img.getdata())
            img_str = ' '
            for idx in range(len(img_flat)):
                if idx != len(img_flat) - 1:
                    img_str += str(img_flat[idx]) + " "
                else:
                    img_str += str(img_flat[idx])

            out_data['pixels'].append(img_str)
            out_data['emotion'].append(emotion)
    df = pd.DataFrame(out_data)
    df.to_csv(out_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    process(os.path.join(args.path,in_path))
