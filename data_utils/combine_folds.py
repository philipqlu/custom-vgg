import pandas as pd
import os
import argparse


def combine(dir='.', file_list, model_name):
    df = pd.DataFrame({'train':[],'val':[]})
    iter = 0
    print file_list
    for file in file_list:
        iter += 1
        temppath = os.path.join(dir, file)
        dftemp = pd.read_csv(temppath)
        if iter == 1:
            df = dftemp
        else:
            df = df.add(dftemp, axis='columns', level=None, fill_value=None)
    df /= len(file_list)
    df.to_csv(model_name, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str)
    parser.add_argument('-o', type=str, default='allfolds.txt')
    args = parser.parse_args()
    files = os.listdir(args.d)
    fold_files = [f for f in files if f[0:4]=='fold']
    combine(args.d, fold_files, args.o)
