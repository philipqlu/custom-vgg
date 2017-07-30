import pandas as pd
import os
from PIL import Image
import argparse
import numpy as np

out_file = 'images_48_48_x2.csv'
in_path = 'cohn-kanade-images'

def main(image_list, data_dir):
    root, directories, files = os.walk(data_dir).next()
    directories = sorted(directories)
    out_data = {'image_url': [],'pixels:'[]}
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
    main(image_list, os.path.join(args.data_dir,in_path))
