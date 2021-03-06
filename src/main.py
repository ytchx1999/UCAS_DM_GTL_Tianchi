import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from torchsummary import summary
import argparse
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.model import ResNet, EYENet
from utils.dataset import EyeDataset


def main():
    # args
    parser = argparse.ArgumentParser(description='Tianchi')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--norm_mean', type=float, default=0.5)
    parser.add_argument('--norm_std', type=float, default=0.2)
    parser.add_argument('--basic_data_dir', type=str, default='../dataset/')
    parser.add_argument('--csv_dir', type=str, default='../data/')
    parser.add_argument('--model_dir', type=str, default='../data/resnet152-b121ed2d.pth')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    args = parser.parse_args()
    print(args)

    pd.set_option('mode.chained_assignment', None)  # pandas no warning

    # train transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

    # test transform
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2])
    ])

    # load dataset
    train_dataset = EyeDataset(data_dir=os.path.join(args.basic_data_dir, 'mix_train'),
                               csv_dir=os.path.join(args.csv_dir, 'train_data.pk'),
                               pkl_dir=os.path.join(args.basic_data_dir, 'pkl_train'),
                               transform=train_transform, mode='train')

    test_dataset = EyeDataset(data_dir=os.path.join(args.basic_data_dir, 'mix_test'),
                              csv_dir=os.path.join(args.csv_dir, 'test_data.pk'),
                              pkl_dir=os.path.join(args.basic_data_dir, 'pkl_test'),
                              transform=test_transform, mode='test')

    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    # load norm info
    with open('../data/norm_info.pk', 'rb') as f:
        norm_info = pickle.load(f)
    # {'preCST': [371.39229340761375, 236.55687441676756],
    #  'VA': [0.6693871866295265, 0.6399039473121558],
    #  'CST': [321.81104921077065, 161.35073145439281]}
    print(norm_info)

    # load submit.csv
    submit = pd.read_csv('../data/submit.csv')
    # patient ID  preCST   VA  continue injection  CST  IRF  SRF  HRF
    print(submit.head())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # model
    # model = EyeNet(
    #     args.model_dir,
    #     num_classes=128,
    #     feat_dim=5,
    #     hidden_dim=128,
    #     output_dim=args.num_classes
    # ).to(device)
    model = EYENet(
        args.model_dir,
        channels=512,
        classes1=64,
        classes2=64,
        base=64,
        feature_dims=1,
        hidden_dim=64,
        output_dim=12
    ).to(device)

    loss_func_reg = nn.MSELoss()  # regression loss func
    loss_func_cls = nn.BCEWithLogitsLoss()  # classification loss func
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # lr adjustment

    # train
    for epoch in range(args.epochs):
        train_loss, train_score = train(train_loader, device, model, loss_func_reg, loss_func_cls, optimizer, norm_info,
                                        epoch, scheduler)
        print(
            f'epoch: {int(epoch):02d}, '
            f'train_loss: {train_loss:.4f}, '
            f'train_score: {train_score:.4f}, '
            # f'val_loss, {val_loss:.4f}, '
            # f'val_acc, {val_acc:.4f} '
        )
    # ResNet-50
    # epoch: 19, train_loss: 0.0745, train_score: 0.4303,

    # ResNet-152
    # epoch: 19, train_loss: 0.0716, train_score: 0.4356,

    # test
    test(test_loader, device, model, norm_info, submit)


def score_reg(out, label, norm_info):
    new_out = out.clone()
    new_label = label.clone()
    # (x - mean) / std = out
    # x = out * std + mean

    # pred
    new_out[:, 0] = new_out[:, 0] * norm_info['preCST'][1] + norm_info['preCST'][0]
    new_out[:, 1] = new_out[:, 1] * norm_info['VA'][1] + norm_info['VA'][0]
    new_out[:, 2] = new_out[:, 2] * norm_info['CST'][1] + norm_info['CST'][0]

    # label
    new_label[:, 0] = new_label[:, 0] * norm_info['preCST'][1] + norm_info['preCST'][0]
    new_label[:, 1] = new_label[:, 1] * norm_info['VA'][1] + norm_info['VA'][0]
    new_label[:, 2] = new_label[:, 2] * norm_info['CST'][1] + norm_info['CST'][0]

    # error rate in [-0.025, 0.025]
    error = new_out - new_label
    error = torch.div(error, new_label)
    error = torch.abs(error)

    mask = error[error <= 0.025]
    return mask.sum().item()  # correct num


def score_cls(out, label):
    new_out = out.clone()
    new_out = torch.sigmoid(new_out)

    # sigmoid(new_out) >= 0.5 --> 1
    # sigmoid(new_out) < 0.5 --> 0
    zero = torch.zeros_like(new_out)
    one = torch.ones_like(new_out)
    new_out = torch.where(new_out >= 0.5, one, new_out)
    new_out = torch.where(new_out < 0.5, zero, new_out)
    new_out = new_out.long()  # long

    new_label = label.clone().long()  # long

    cnt = torch.eq(new_out, new_label).sum().item()  # correct num
    return float(cnt)


def train(train_loader, device, model, loss_func_reg, loss_func_cls, optimizer, norm_info, epoch, scheduler):
    model.train()
    tot_loss = 0
    score = 0
    tot_train = 0
    for i, data in enumerate(train_loader):
        patient_ids, pre_img, after_img, pre_pkl, after_pkl, img_feats, img_labels = data
        pre_img, after_img, pre_pkl, after_pkl, img_feats, img_labels = pre_img.to(device), after_img.to(
            device), pre_pkl.to(device), after_pkl.to(device), img_feats.to(device), img_labels.to(device)

        out = model(pre_img, after_img, img_feats, pre_pkl, after_pkl)
        # regression loss
        loss_reg = loss_func_reg(out[:, :3], img_labels[:, :3])
        # classification loss
        loss_cls = loss_func_cls(out[:, 3:], img_labels[:, 3:])
        # sum
        loss = loss_reg + loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()

        # calculate score
        correct_num = (score_cls(out[:, 3:7], img_labels[:, 3:7]) + score_reg(out[:, :3], img_labels[:, :3], norm_info))
        score += correct_num

        tot_train += img_labels.size(0)

        print(
            f'epoch: {int(epoch):02d}, '
            f'iter: {i:02d}, '
            f'batch_loss: {loss:.4f}, '
            f'batch_correct_num: {int(correct_num):4d}, '
            f'tot_score: {(score / (tot_train * 7)):.4f}, '
        )
        # print(loss_reg.item(), loss_cls.item())

    scheduler.step()
    print(tot_train)

    return tot_loss / tot_train, score / (tot_train * 7)


@torch.no_grad()
def test(test_loader, device, model, norm_info, submit):
    model.eval()

    # mkdir ../outputs
    if not os.path.exists('../outputs'):
        os.makedirs('../outputs', exist_ok=True)

    for i, data in tqdm(enumerate(test_loader)):
        patient_ids, pre_img, after_img, pre_pkl, after_pkl, img_feats = data
        pre_img, after_img, pre_pkl, after_pkl, img_feats = pre_img.to(device), after_img.to(device), pre_pkl.to(
            device), after_pkl.to(device), img_feats.to(device)

        out = model(pre_img, after_img, img_feats, pre_pkl, after_pkl)

        # regression transform
        new_reg = out[:, :3].clone()

        new_reg[:, 0] = new_reg[:, 0] * norm_info['preCST'][1] + norm_info['preCST'][0]
        new_reg[:, 1] = new_reg[:, 1] * norm_info['VA'][1] + norm_info['VA'][0]
        new_reg[:, 2] = new_reg[:, 2] * norm_info['CST'][1] + norm_info['CST'][0]

        # classification transform
        new_cls = out[:, 3:7].clone()
        new_cls = torch.sigmoid(new_cls)

        zero = torch.zeros_like(new_cls)
        one = torch.ones_like(new_cls)
        new_cls = torch.where(new_cls >= 0.5, one, new_cls)
        new_cls = torch.where(new_cls < 0.5, zero, new_cls)
        new_cls = new_cls.long()

        # update submit by patient id index
        for out_index, patient_id in enumerate(patient_ids):
            # get index from submit.csv
            csv_index = submit[submit['patient ID'] == patient_id].index.tolist()[0]

            # preCST VA CST
            submit['preCST'][csv_index] = new_reg[out_index, 0]
            submit['VA'][csv_index] = new_reg[out_index, 1]
            submit['CST'][csv_index] = new_reg[out_index, 2]

            # continue injection IRF  SRF  HRF
            submit['continue injection'][csv_index] = new_cls[out_index][0]
            submit['IRF'][csv_index] = new_cls[out_index][1]
            submit['SRF'][csv_index] = new_cls[out_index][2]
            submit['HRF'][csv_index] = new_cls[out_index][3]

    # write submit to ../outputs/submit_xxxxx.csv
    submit.to_csv(os.path.join('../outputs/', f'submit_{time.strftime("%Y-%m-%d", time.localtime())}.csv'), index=False)


# main
if __name__ == '__main__':
    main()
