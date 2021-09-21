import cv2
import os
import numpy as np

if __name__ == "__main__":
    # add the pre images together and both the pro images
    root = 'pro_train'
    for Set in os.listdir(root):
        if Set[:-1] == 'trainset':
            for user in os.listdir(os.path.join('pro_train', Set)):
                path = os.path.join(f'pro_train/{Set}', user)
                pre_img = 0
                pro_img = 0
                for pic in os.listdir(path):
                    currpath = os.path.join(path, pic)
                    dispath = 'mix_train/' + currpath[6:-20]
                    if not os.path.exists(os.path.abspath(dispath)[:-20]):
                        os.makedirs(os.path.abspath(dispath)[:-20])
                    img = cv2.imread(currpath)
                    if pic[-8] == '1':
                        if pre_img == 0:
                            pre_img = img
                        else:
                            pre_img = cv2.add(pre_img, img)
                    if pic[-8] == '2':
                        if pro_img == 0:
                            pro_img = img
                        else:
                            pro_img = cv2.add(pro_img, img)
                    dispath1 = os.path.join(dispath, 'pre.jpg')
                    dispath2 = os.path.join(dispath, 'pro.jpg')
                    cv2.imwrite(dispath1, pre_img)
                    cv2.imwrite(dispath2, pro_img)
