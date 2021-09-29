import numpy as np
import pandas as pd
import torch
import pickle
from torchsummary import summary
import argparse
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.model import ResNet
from utils.dataset import EyeDataset


def main():
    # args
    parser = argparse.ArgumentParser(description='Tianchi')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--norm_mean', type=float, default=0.5)
    parser.add_argument('--norm_std', type=float, default=0.2)
    parser.add_argument('--basic_data_dir', type=str, default='../dataset/')
    parser.add_argument('--csv_dir', type=str, default='../data/')
    parser.add_argument('--model_dir', type=str, default='../data/resnet18-5c106cde.pth')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    print(args)

    # transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

    # load dataset
    train_dataset = EyeDataset(data_dir=os.path.join(args.basic_data_dir, 'mix_train'),
                               csv_dir=os.path.join(args.csv_dir, 'train_data.pk'),
                               transform=transform, mode='train')
    # print(train_dataset.data_info[0])

    test_dataset = EyeDataset(data_dir=os.path.join(args.basic_data_dir, 'mix_test'),
                              csv_dir=os.path.join(args.csv_dir, 'test_data.pk'),
                              transform=transform, mode='test')
    # print(test_dataset.data_info[0])

    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    # load train data
    # with open('../data/train_data.pk', 'rb') as f:
    #     train_id_index, train_feats, train_labels_dict = pickle.load(f)
    # print(
    #     'len train_id_index=', len(train_id_index),
    #     'train_feats.shape=', train_feats.shape,
    #     'len train_labels', len(train_labels_dict)
    # )
    #
    # # load test data
    # with open('../data/test_data.pk', 'rb') as f:
    #     test_id_index, test_feats = pickle.load(f)
    # print(
    #     'len test_id_index=', len(test_id_index),
    #     'test_feats.shape=', test_feats.shape
    # )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # a simple test for dataloader (load a batch) -- success!
    # torch.Size([16, 3, 224, 224])
    # torch.Size([16, 3, 224, 224])
    # torch.Size([16, 5])
    # torch.Size([16, 12])
    for i, data in enumerate(train_loader):
        pre_img, after_img, img_feats, img_labels = data
        pre_img, after_img, img_feats, img_labels = pre_img.to(device), after_img.to(device), img_feats.to(
            device), img_labels.to(device)

        if pre_img != None:
            print(pre_img.shape)
        if after_img != None:
            print(after_img.shape)
        print(img_feats.shape)
        print(img_labels.shape)

        break

    # model
    resnet_18 = ResNet(args.model_dir, num_classes=args.num_classes).to(device)
    summary(resnet_18, input_size=(3, 224, 224))


# main
if __name__ == '__main__':
    main()
