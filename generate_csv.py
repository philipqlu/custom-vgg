import pandas as pd
import os
from PIL import Image
import argparse

# no contempt
FLAGS = None
def main(image_list):
    root, directories, files = os.walk(FLAGS.data_dir).next()
    directories = sorted(directories)
    pixels_list, image_names = [], []

    for i in range(len(directories)):
        temppath = FLAGS.data_dir + directories[i]
        # go through each subject's folders in image directory
        for subfolder in os.listdir(temppath):
            # ignore hidden folders in image folders
            if subfolder.startswith('.'):
    	        continue
            # get path of image directory
            temppath2 = temppath + '/' + subfolder
            templist = sorted(os.listdir(temppath2))
            image_file = templist[-1]
            # check if it's part of the dataset
            if image_file.rstrip('.png') in image_list:
                img = Image.open(temppath2+'/'+image_file)
                img = img.resize((128,98), Image.ANTIALIAS)
                img_shape = img.size
                img_flat = list(img.getdata())
                img_str = ''
                for idx in range(len(img_flat)):
                    if idx != len(img_flat) - 1:
                        img_str += str(img_flat[idx]) + " "
                    else:
                        img_str += str(img_flat[idx])
                pixels_list.append(img_str)
                image_names.append(image_file)

    print len(pixels_list), len(image_names)
    df_result = pd.DataFrame({'image_url':image_names, 'pixels':pixels_list})

    # write to file
    df_result.to_csv('compressed_images.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='tmp/data/',
                      help='Directory of the images')
    FLAGS = parser.parse_args()
    df = pd.read_csv('image_list.csv')
    image_list = list(df.image_url.str.rstrip('.jpg'))
    print image_list[0]
    main(image_list)
