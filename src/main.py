import numpy as np
import pandas as pd
import torch
import pickle
from torchsummary import summary
import argparse

from models.model import ResNet


def main():
    # args
    parser = argparse.ArgumentParser(description='Tianchi')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--norm_mean', type=float, default=0.5)
    parser.add_argument('--norm_std', type=float, default=0.2)
    parser.add_argument('--basic_data_dir', type=str, default='../data/')
    parser.add_argument('--model_dir', type=str, default='../data/resnet18-5c106cde.pth')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    print(args)

    # load train data
    with open('../data/train_data.pk', 'rb') as f:
        train_id_index, train_feats, train_labels_dict = pickle.load(f)
    print(
        'len train_id_index=', len(train_id_index),
        'train_feats.shape=', train_feats.shape,
        'len train_labels', len(train_labels_dict)
    )

    # load test data
    with open('../data/test_data.pk', 'rb') as f:
        test_id_index, test_feats = pickle.load(f)
    print(
        'len test_id_index=', len(test_id_index),
        'test_feats.shape=', test_feats.shape
    )

    # print(torch.cat([train_labels_dict['preIRF'], train_labels_dict['preSRF']], dim=1))

    # model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    resnet_18 = ResNet(args.model_dir, num_classes=args.num_classes).to(device)
    summary(resnet_18, input_size=(3, 224, 224))


# main
if __name__ == '__main__':
    main()
