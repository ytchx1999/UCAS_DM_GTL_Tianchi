from posixpath import basename
from types import DynamicClassAttribute
import cv2
import os
import numpy as np

if __name__ == "__main__":
    # preprocess the images and put the processed images to the new folder
    root = '../../dataset/split'
    for trainval in os.listdir(root):
        root = '../../dataset/split'
        root = os.path.join(root, trainval)
        if os.path.isdir(root):
            for Set in os.listdir(root):
                # print(Set)
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
                        if isinstance(la, int) and pic[9] == 'L' and pic[11] == '2':
                            la = cv2.imread(currpath)
                            la = np.array(la, dtype=float)
                            n2 += 1
                        elif isinstance(lb, int) and pic[9] == 'L' and pic[11] == '1':
                            lb = cv2.imread(currpath)
                            lb = np.array(lb, dtype=float)
                            n1 += 1
                        elif isinstance(rb, int) and pic[9] == 'R' and pic[11] == '1':
                            rb = cv2.imread(currpath)
                            rb = np.array(rb, dtype=float)
                            n3 += 1
                        elif isinstance(ra, int) and pic[9] == 'R' and pic[11] == '2':
                            ra = cv2.imread(currpath)
                            ra = np.array(ra, dtype=float)
                            n4 += 1
                        if pic[9] == 'L' and pic[11] == '2':
                            la += np.array(cv2.imread(currpath), dtype=float)
                            n2 += 1
                        elif pic[9] == 'L' and pic[11] == '1':
                            lb += np.array(cv2.imread(currpath), dtype=float)
                            n1 += 1
                        elif pic[9] == 'R' and pic[11] == '2':
                            rb += np.array(cv2.imread(currpath), dtype=float)
                            n3 += 1
                        elif pic[9] == 'R' and pic[11] == '2':
                            ra += np.array(cv2.imread(currpath), dtype=float)
                            n4 += 1
                        dispath = os.path.join('../../dataset', f'mix_{trainval}/' + currpath[6:])
                        print(os.path.abspath(dispath)[:-20])
                        print(dispath)
                    if not os.path.exists(os.path.abspath(dispath)[:-20]):
                        os.makedirs(os.path.abspath(dispath)[:-20])
                    if n1 > 0:
                        img = (lb / n1).astype(np.int8)
                        cv2.imwrite(os.path.join(dispath[:-20], user + 'L_1.jpg'), img)
                    if n2 > 0:
                        img = (la / n2).astype(np.int8)
                        cv2.imwrite(os.path.join(dispath[:-20], user + 'L_2.jpg'), img)
                    if n3 > 0:
                        img = (rb / n3).astype(np.int8)
                        cv2.imwrite(os.path.join(dispath[:-20], user + 'R_1.jpg'), img)
                    if n4 > 0:
                        img = (ra / n4).astype(np.int8)
                        cv2.imwrite(os.path.join(dispath[:-20], user + 'R_2.jpg'), img)
