from posixpath import basename
from types import DynamicClassAttribute
import cv2
import os
import numpy as np

if __name__ == "__main__":
    x = 100
    # preprocess the images and put the processed images to the new folder
    root = '../../dataset/split'
    for trainval in os.listdir(root):
        root = '../../dataset/split'
        root = os.path.join(root, trainval)
        if os.path.isdir(root) and trainval == 'test':
            for Set in os.listdir(root):
                if not os.path.isdir(os.path.join(root, Set)):
                    continue
                for user in os.listdir(os.path.join(root, Set)):
                    path = os.path.join(f'{root}/{Set}', user)
                    lb, la, rb, ra = 0, 0, 0, 0
                    n1, n2, n3, n4 = 0, 0, 0, 0
                    if not os.path.isdir(path):
                        continue
                    for pic in os.listdir(path):
                        currpath = os.path.join(path, pic)
                        if currpath[-4:] != '.jpg' or cv2.imread(currpath).shape[1] != 772:
                            continue
                        flag = pic.split('.')[0][-3:]
                        if flag == 'L_1':
                            n1 +=1
                        elif flag == 'L_2':
                            n2 += 1
                        elif flag == 'L_3':
                            n3 += 1
                        elif flag == 'L_4':
                            n4 += 1
                    x = min(x, n1)
                    x = min(x, n2)
                    x = min(x, n3)
                    x = min(x, n4)