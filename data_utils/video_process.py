import cv2
import os
import xml.etree.ElementTree as ET
import argparse

in_path = 'Sessions'
out_path = 'mmidata'

def emotion_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for child in root:
        if child.get('Name') == 'Emotion':
            return child.get('Value')
    return -1

def get_files(filelist):
    x, v = '',''
    for fl in filelist:
        name, ext = os.path.splitext(fl)
        if ext == '.avi':
            v = fl
        if ext == '.xml' and name != 'session':
            x = fl
    return x,v

def process(file_directory):
    subfolders = os.listdir(file_directory)
    print subfolders
    for folder in subfolders:
        if folder[0] == '.':
            continue
        temppath = os.path.join(file_directory, folder)
        xml_file, video_file = get_files(os.listdir(temppath))
        if len(xml_file) == 0:
            print 'missing xml file'
        if len(video_file) == 0:
            print 'missing avi file'
        else:
            v_name = video_file.rstrip('.avi')
            video_file = os.path.join(temppath,video_file)
            vidcap = cv2.VideoCapture(video_file)
            success, image = vidcap.read()
            count = 0
            while success and count < 5:
              vidcap.set(cv2.CAP_PROP_POS_MSEC,400 + 400*count)
              success,image = vidcap.read()
              print('Read a new frame: ', success)
              outpath2 = os.path.join(out_path,v_name)
              if not os.path.exists(outpath2):
                  os.mkdir(outpath2)
              tempout = os.path.join(outpath2,v_name+"%.2d.jpg" % count)
              cv2.imwrite(tempout, image)
              emotion = emotion_xml(os.path.join(temppath,xml_file))
              with open(os.path.join(outpath2,'emot.txt'), 'wa+') as f:
                  f.write(emotion)
              count += 1

if __name__ == '__main__':
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    process(os.path.join(args.path,in_path))
