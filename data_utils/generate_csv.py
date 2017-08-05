import pandas as pd
import os
from PIL import Image
import argparse
import numpy as np

out_file = 'mmi_48_48.csv'
in_path = 'mmidata'

def mmi_extract(file_list):
    image_files, emotion_file = [], ''
    for f in file_list:
        if f[-3:] == 'jpg':
            image_files.append(f)
        elif f[-3:] == 'txt':
            emotion_file = f
    return image_files, emotion_file

def mmi(data_dir):
    root, directories, files = os.walk(data_dir).next()
    directories = sorted(directories)
    out_data = {'image_url': [],'pixels':[],'emotion':[]}
    for i in range(len(directories)):
        temppath = os.path.join(data_dir, directories[i])
        image_files, emotion_file = mmi_extract(os.listdir(temppath))
        with open(os.path.join(temppath,emotion_file),'r+') as f:
            emotion = int(f.readline())
        for image_file in image_files:
            img_path = os.path.join(temppath,image_file)
            img = Image.open(img_path).convert('L')
            img = img.resize((48,48), Image.ANTIALIAS)
            img_flat = np.asarray(img.getdata())
            img_str = ''
            for idx in range(len(img_flat)):
                if idx != len(img_flat) - 1:
                    img_str += str(img_flat[idx]) + " "
                else:
                    img_str += str(img_flat[idx])
            out_data['pixels'].append(img_str)
            out_data['image_url'].append(image_file)
            out_data['emotion'].append(emotion)
            i += 1
    df1 = pd.DataFrame(out_data)
    df_result = pd.concat([df1])
    df_result.to_csv(out_file, index=False)

def main(image_list, data_dir):
    root, directories, files = os.walk(data_dir).next()
    directories = sorted(directories)
    out_data = {'image_url': [],'pixels':[]}
    for i in range(len(directories)):
        temppath = os.path.join(data_dir, directories[i])
        # go through each subject's folders in image directory
        for subfolder in sorted(os.listdir(temppath)):
            # ignore hidden folders in image folders
            if subfolder.startswith('.'):
    	        continue
            templist = sorted(os.listdir(os.path.join(temppath,subfolder)))
            image_files = [templist[-1]]
            # check if it's part of the dataset
            if image_files[0].rstrip('.png') in image_list:
                i = 0
                for image_file in image_files:
                    img_path = os.path.join(temppath2,image_file)
                    img = Image.open(img_path).convert('L')
                    img = img.resize((48,48), Image.ANTIALIAS)
                    img_flat = np.asarray(img.getdata())
                    img_str = ''
                    for idx in range(len(img_flat)):
                        if idx != len(img_flat) - 1:
                            img_str += str(img_flat[idx]) + " "
                        else:
                            img_str += str(img_flat[idx])
                    out_data['pixels'].append(img_str)
                    out_data['image_url'].append(image_file)
                    i += 1

    df1 = pd.DataFrame(out_data)
    df_result = pd.concat([df1])
    df_result.to_csv(out_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='tmp/data/',
                      help='Directory of the images')
    args = parser.parse_args()
    df = pd.read_csv('image_list.csv')
    image_list = list(df.image_url.str.rstrip('.jpg'))
    print image_list[0]
    # main(image_list, os.path.join(args.data_dir,in_path))
    mmi(args.data_dir)
