import cv2
import os
import numpy as np

if __name__ == "__main__":
    root = 'train'
    for Set in os.listdir(root):
        print(Set)
        if Set[:-1] == 'trainset':
            for user in os.listdir(os.path.join('train', Set)):
                for pic in os.listdir(os.path.join('train/trainset1', '0000-0230')):
                    print(pic)
                    img = cv2.imread(os.path.join('train/trainset1/0000-0000', '0000-0000L_1000.jpg'))
                    res = img[:500, 492:, :]
                    cv2.imwrite('res.jpg', res)
                    break
