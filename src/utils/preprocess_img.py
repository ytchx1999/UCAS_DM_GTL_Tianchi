from posixpath import basename
import cv2
import os
import numpy as np

if __name__ == "__main__":
    # preprocess the images and put the processed images to the new folder
    root = '../../dataset'
    for trainval in os.listdir(root):
        root = '../../dataset'
        root = os.path.join(root, trainval)
        if os.path.isdir(root):
            for Set in os.listdir(root):
                # print(Set)
                if os.path.isdir(os.path.join(root, Set)):
                    for user in os.listdir(os.path.join(root, Set)):
                        path = os.path.join(f'{root}/{Set}', user)
                        if os.path.isdir(path):
                            for pic in os.listdir(path):
                                currpath = os.path.join(path, pic)
                                dispath = os.path.join('../../dataset', f'pro_{trainval}/' + currpath[6:])
                                print(os.path.abspath(dispath)[:-20])
                                print(dispath)
                                if not os.path.exists(os.path.abspath(dispath)[:-20]):
                                    os.makedirs(os.path.abspath(dispath)[:-20])
                                img = cv2.imread(currpath)
                                res = img[:500, 492:, :]
                                cv2.imwrite(dispath, res)
