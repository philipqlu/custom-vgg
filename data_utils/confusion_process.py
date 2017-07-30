from __future__ import division
import os
import argparse
import numpy as np


def process(files, out='result.txt'):
    conf_matrix = np.zeros((6,6))
    iterations = 0
    for file_name in files:
        if not os.path.exists(file_name):
            print file_name + ' not found'
        else:
            iterations += 1
            with open(file_name, 'rb') as f:
                rows = f.readlines()
                print rows
                for i in range(len(rows)):
                    el_list = map(int, rows[i][:-1].split(','))
                    conf_matrix[i] += np.asarray(el_list)

    conf_matrix /= iterations
    np.savetxt(out, conf_matrix, fmt='%.2f', delimiter=',')

if __name__ == "__main__":
    files = []
    for i in range(10):
        files.append('confusion'+str(i+1)+'.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, default='result.txt')
    args = parser.parse_args()
    process(files, args.o)
