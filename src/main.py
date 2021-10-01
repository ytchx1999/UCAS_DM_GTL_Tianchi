import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from torchsummary import summary
import argparse
import os
import time
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.model import ResNet, EyeNet
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
    parser.add_argument('--model_dir', type=str, default='../data/resnet18-5c106cde.pth')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    print(args)

    pd.set_option('mode.chained_assignment', None)

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

    with open('../data/norm_info.pk', 'rb') as f:
        norm_info = pickle.load(f)
    # {'preCST': [371.39229340761375, 236.55687441676756],
    #  'VA': [0.6693871866295265, 0.6399039473121558],
    #  'CST': [321.81104921077065, 161.35073145439281]}
    print(norm_info)

    submit = pd.read_csv('../data/submit.csv')
    # patient ID  preCST   VA  continue injection  CST  IRF  SRF  HRF
    print(submit.head())

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

    # for i, data in enumerate(train_loader):
    #     patient_ids, pre_img, after_img, img_feats, img_labels = data
    #     pre_img, after_img, img_feats, img_labels = pre_img.to(device), after_img.to(
    #         device), img_feats.to(device), img_labels.to(device)
    #
    #     # if pre_img != None:
    #     print(patient_id)
    #     print(pre_img.shape)
    #     # if after_img != None:
    #     print(after_img.shape)
    #     print(img_feats.shape)
    #     print(img_labels.shape)
    #
    #     break

    # for i, data in enumerate(test_loader):
    #     patient_ids, pre_img, after_img, img_feats = data
    #     pre_img, after_img, img_feats = pre_img.to(device), after_img.to(device), img_feats.to(device)
    #
    #     print(patient_id)
    #     print(pre_img.shape)
    #     print(after_img.shape)
    #     print(img_feats.shape)

    #########################################
    # model
    # resnet_18 = ResNet(args.model_dir, num_classes=args.num_classes).to(device)
    # summary(resnet_18, input_size=(3, 224, 224))
    model = EyeNet(
        args.model_dir,
        num_classes=16,
        feat_dim=5,
        hidden_dim=16,
        output_dim=args.num_classes
    ).to(device)

    loss_func_reg = nn.MSELoss()
    loss_func_cls = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, train_score = train(train_loader, device, model, loss_func_reg, loss_func_cls, optimizer, norm_info)
        print(
            f'epoch: {epoch:02d}, '
            f'train_loss: {train_loss:.4f}, '
            f'train_acc: {train_score:.4f}, '
            # f'val_loss, {val_loss:.4f}, '
            # f'val_acc, {val_acc:.4f} '
        )

    test(test_loader, device, model, norm_info, submit)


def score_reg(out, label, norm_info):
    new_out = out.clone()
    new_label = label.clone()
    new_out[:, 0] = new_out[:, 0] * norm_info['preCST'][1] + norm_info['preCST'][0]
    new_out[:, 1] = new_out[:, 1] * norm_info['VA'][1] + norm_info['VA'][0]
    new_out[:, 2] = new_out[:, 2] * norm_info['CST'][1] + norm_info['CST'][0]

    new_label[:, 0] = new_label[:, 0] * norm_info['preCST'][1] + norm_info['preCST'][0]
    new_label[:, 1] = new_label[:, 1] * norm_info['VA'][1] + norm_info['VA'][0]
    new_label[:, 2] = new_label[:, 2] * norm_info['CST'][1] + norm_info['CST'][0]

    error = new_out - new_label
    error = torch.div(error, new_label)
    error = torch.abs(error)

    mask = error[error <= 0.025]
    return mask.sum().item()


def score_cls(out, label):
    cnt = 0
    cnt = torch.eq(out, label)
    # for i in range(out.shape[0]):
    #     for j in range(out.shape[1]):
    #         if out[i][j] == label[i][j]:
    #             cnt += 1
    return cnt


def train(train_loader, device, model, loss_func_reg, loss_func_cls, optimizer, norm_info):
    model.train()
    tot_loss = 0
    score = 0
    tot_train = 0
    for i, data in enumerate(train_loader):
        patient_ids, pre_img, after_img, img_feats, img_labels = data
        pre_img, after_img, img_feats, img_labels = pre_img.to(device), after_img.to(device), img_feats.to(
            device), img_labels.to(device)

        out = model(pre_img, after_img, img_feats)
        loss_reg = loss_func_reg(out[:, :3], img_labels[:, :3])
        loss_cls = loss_func_cls(out[:, 3:], img_labels[:, 3:])

        loss = loss_reg + loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()

        score += (score_cls(out[:, 3:7], img_labels[:, 3:7]) + score_reg(out[:, :3], img_labels[:, :3], norm_info))
        # y_pred = out.argmax(dim=1)
        # correct = (y_pred == label).sum().item()
        # tot_correct += correct
        tot_train += img_labels.size(0)

    return tot_loss / tot_train, score


@torch.no_grad()
def test(test_loader, device, model, norm_info, submit):
    model.eval()
    if not os.path.exists('../outputs'):
        os.makedirs('../outputs', exist_ok=True)
    for i, data in enumerate(test_loader):
        patient_ids, pre_img, after_img, img_feats = data
        pre_img, after_img, img_feats = pre_img.to(device), after_img.to(device), img_feats.to(device)

        out = model(pre_img, after_img, img_feats)

        new_reg = out[:, :3].clone()

        new_reg[:, 0] = new_reg[:, 0] * norm_info['preCST'][1] + norm_info['preCST'][0]
        new_reg[:, 1] = new_reg[:, 1] * norm_info['VA'][1] + norm_info['VA'][0]
        new_reg[:, 2] = new_reg[:, 2] * norm_info['CST'][1] + norm_info['CST'][0]

        new_cls = out[:, 3:7].clone()
        new_cls = torch.sigmoid(new_cls)

        zero = torch.zeros_like(new_cls)
        one = torch.ones_like(new_cls)
        new_cls = torch.where(new_cls >= 0.5, one, new_cls)
        new_cls = torch.where(new_cls < 0.5, zero, new_cls)
        new_cls = new_cls.long()

        for out_index, patient_id in enumerate(patient_ids):
            csv_index = submit[submit['patient ID'] == patient_id].index.tolist()[0]
            submit['preCST'][csv_index] = new_reg[out_index, 0]
            submit['VA'][csv_index] = new_reg[out_index, 1]
            submit['CST'][csv_index] = new_reg[out_index, 2]

            # continue injection IRF  SRF  HRF
            submit['continue injection'][csv_index] = new_cls[out_index][0]
            submit['IRF'][csv_index] = new_cls[out_index][1]
            submit['SRF'][csv_index] = new_cls[out_index][2]
            submit['HRF'][csv_index] = new_cls[out_index][3]

    submit.to_csv(os.path.join('../outputs/', f'submit_{time.strftime("%Y-%m-%d", time.localtime())}.csv'), index=False)

    # return tot_loss / tot_train, score


# main
if __name__ == '__main__':
    main()
