from posixpath import basename
import cv2
import os
import numpy as np


if __name__ == "__main__":
    # preprocess the images and put the processed images to the new folder
    root = 'train'
    for Set in os.listdir(root):
        # print(Set)
        if Set[:-1] == 'trainset':
            for user in os.listdir(os.path.join('train', Set)):
                path = os.path.join(f'train/{Set}', user)
                for pic in os.listdir(path):
                    currpath = os.path.join(path, pic)
                    dispath = 'pro_train/' + currpath[6:]
                    if not os.path.exists(os.path.abspath(dispath)[:-20]):
                        os.makedirs(os.path.abspath(dispath)[:-20])
                    img = cv2.imread(currpath)
                    res = img[:500,492:,:]
                    cv2.imwrite(dispath, res)
