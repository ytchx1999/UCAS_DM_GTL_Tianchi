from numpy.lib.arraysetops import isin
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
import numpy as np
import cv2
import os
import torch
import pickle
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import wget


def read_img(path):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    lb, la, rb, ra = [], [], [], []
    n1, n2, n3, n4 = 0, 0, 0, 0
    if not os.path.isdir(path):
        return
    for pic in os.listdir(path):
        currpath = os.path.join(path, pic)
        if currpath[-4:] != '.jpg' or cv2.imread(currpath).shape[1] != 772:
            continue
        if isinstance(la, int) and pic[9] == 'L' and pic[11] == '2':
            la = Image.open(currpath).convert('RGB')
            la = transform(la)
            n2 += 1
        elif isinstance(lb, int) and pic[9] == 'L' and pic[11] == '1':
            lb = Image.open(currpath).convert('RGB')
            lb = transform(lb)
            n1 += 1
        elif isinstance(rb, int) and pic[9] == 'R' and pic[11] == '1':
            rb = Image.open(currpath).convert('RGB')
            rb = transform(rb)
            n3 += 1
        elif isinstance(ra, int) and pic[9] == 'R' and pic[11] == '2':
            ra = Image.open(currpath).convert('RGB')
            ra = transform(ra)
            n4 += 1
        if pic[9] == 'L' and pic[11] == '2':
            la.append(transform(Image.open(currpath).convert('RGB')))
            n2 += 1
        elif pic[9] == 'L' and pic[11] == '1':
            lb.append(transform(Image.open(currpath).convert('RGB')))
            n1 += 1
        elif pic[9] == 'R' and pic[11] == '2':
            rb.append(transform(Image.open(currpath).convert('RGB')))
            n3 += 1
        elif pic[9] == 'R' and pic[11] == '2':
            ra.append(transform(Image.open(currpath).convert('RGB')))
            n4 += 1
    if n1 > 0:
        lb = torch.stack(lb)
    if n2 > 0:
        la = torch.stack(la)
    if n3 > 0:
        rb = torch.stack(rb)
    if n4 > 0:
        ra = torch.stack(ra)
    return lb, la, rb, ra
                    

# path = '../../dataset/split'
class FeatureExtraction:
    def __init__(self, path):
        super(FeatureExtraction, self).__init__()
        self.model = ResNet()
        for trainval in os.listdir(path):
            root = os.path.join(path, trainval)
            if os.path.isdir(root):
                for Set in os.listdir(root):
                    if not os.path.isdir(os.path.join(root, Set)):
                        continue
                    for user in os.listdir(os.path.join(root, Set)):
                        dirpath = os.path.join(f'{root}/{Set}', user)
                        # 返回left_before, left_after, right_before, right_after
                        left_before, left_after, right_before, right_after = read_img(dirpath)
                        if isinstance(left_before, torch.Tensor):
                            left_before = self.model(left_before)
                            left_before = torch.sum(left_before, dim=0)
                            left_before = torch.squeeze(left_before, dim=0)
                            pickle.dump(left_before, os.path.join(dirpath, user+'L_1.pkl'))
                        if isinstance(left_after, torch.Tensor):
                            left_after = self.model(left_after)
                            left_after = torch.sum(left_after, dim=0)
                            left_after = torch.squeeze(left_after, dim=0)
                            pickle.dump(left_after, os.path.join(dirpath, user+'L_2.pkl'))
                        if isinstance(right_before, torch.Tensor):
                            right_before = self.model(right_before)
                            right_before = torch.sum(right_before, dim=0)
                            right_before = torch.squeeze(right_before, dim=0)
                            pickle.dump(right_before, os.path.join(dirpath, user+'R_1.pkl'))
                        if isinstance(right_after, torch.Tensor):
                            right_after = self.model(right_after)
                            right_after = torch.sum(right_after, dim=0)
                            right_after = torch.squeeze(right_after, dim=0)
                            pickle.dump(right_after, os.path.join(dirpath, user+'R_2.pkl'))


class ResNet(nn.Module):
    def __init__(self, pretrain_dir, num_classes=1000):
        '''
        :param pretain_dir: location of pretrained model resnet-18 (../../data/resnet18-5c106cde.pth)
        :param num_classes: label dims
        '''
        super(ResNet, self).__init__()
        self.pretrain_dir = pretrain_dir
        self.num_classes = num_classes
        self.resnet = torchvision.models.resnet50()

        # download or use cached model
        if not os.path.exists(self.pretrain_dir):
            # download
            # https://download.pytorch.org/models/resnet18-5c106cde.pth
            url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            wget.download(url, "../../data/resnet50-19c8e357.pth")

        # load state dict (params) of the model
        state_dict_load = torch.load(self.pretrain_dir, map_location='cpu')
        self.resnet.load_state_dict(state_dict_load)

        # modify the last FC layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-10])

    def forward(self, x):
        x = self.resnet(x)
        return x

