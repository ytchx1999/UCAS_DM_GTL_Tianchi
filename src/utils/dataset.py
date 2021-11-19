import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import pickle
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


# data_dir = ../../dataset/mix_train/
# 0000-0000/0000-0000L_1.jpg
class EyeDataset(Dataset):
    def __init__(self, data_dir, csv_dir, pkl_dir, transform=None, mode='train'):
        super(EyeDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.pkl_dir = pkl_dir
        self.mode = mode
        self.data_info = self.get_data(self.data_dir, self.csv_dir, self.pkl_dir, self.mode)
        self.transform = transform

    def __getitem__(self, item):
        '''
        :param item: index
        :return: if train, return pre_img, after_img, img_feats, img_labels
                 else, return pre_img, after_img, img_feats (no labels)
        '''
        if self.mode == 'train':
            patient_id, pre_img_path, after_img_path, pre_pkl_path, after_pkl_path, img_feats, img_labels = \
                self.data_info[item]
        else:
            patient_id, pre_img_path, after_img_path, pre_pkl_path, after_pkl_path, img_feats = self.data_info[item]

        if pre_img_path != '':
            pre_img = Image.open(pre_img_path).convert('RGB')
            with open(pre_pkl_path, 'rb') as f:
                pre_pkl = pickle.load(f)
        else:
            pre_img = None
            pre_pkl = None
        if after_img_path != '':
            after_img = Image.open(after_img_path).convert('RGB')
            with open(after_pkl_path, 'rb') as f:
                after_pkl = pickle.load(f)
        else:
            after_img = None
            after_pkl = None

        if self.transform != None:
            if pre_img_path != '':
                pre_img = self.transform(pre_img)
            if after_img_path != '':
                after_img = self.transform(after_img)

        if self.mode != 'train':
            return patient_id, pre_img, after_img, pre_pkl, after_pkl, img_feats
        return patient_id, pre_img, after_img, pre_pkl, after_pkl, img_feats, img_labels

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_data(data_dir, csv_dir, pkl_dir, mode):
        with open(csv_dir, 'rb') as f:
            if mode == 'train':
                id_index, feats, labels = pickle.load(f)
            else:
                id_index, feats = pickle.load(f)

        data_info = []
        for _, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                if (sub_dir[:9] + 'L' in id_index) or (sub_dir[:9] + 'R' in id_index):
                    l_img_index = np.argwhere(id_index == sub_dir[:9] + 'L')
                    r_img_index = np.argwhere(id_index == sub_dir[:9] + 'R')
                else:
                    continue

                # judge l_img_index is None or not
                if l_img_index.shape[0] != 0:
                    l_patient_id = id_index[l_img_index[0][0]]
                    # slice: shape(n, )
                    l_img_feats = torch.from_numpy(feats[l_img_index[0][0]]).float()
                    if mode == 'train':
                        l_img_labels = torch.from_numpy(labels[l_img_index[0][0]]).float()

                if r_img_index.shape[0] != 0:
                    r_patient_id = id_index[r_img_index[0][0]]
                    r_img_feats = torch.from_numpy(feats[r_img_index[0][0]]).float()
                    if mode == 'train':
                        r_img_labels = torch.from_numpy(labels[r_img_index[0][0]]).float()

                # pre, after img_path
                l_pre_img_path = ''
                l_after_img_path = ''
                r_pre_img_path = ''
                r_after_img_path = ''

                l_pre_pkl_path = ''
                l_after_pkl_path = ''
                r_pre_pkl_path = ''
                r_after_pkl_path = ''

                for _, _, files in os.walk(os.path.join(data_dir, sub_dir)):
                    for file in files:
                        # print(file)
                        if file[9] == 'L' and file[-5] == '1':
                            l_pre_img_path = os.path.join(data_dir, sub_dir, file)
                            l_pre_pkl_path = os.path.join(pkl_dir, sub_dir, file[:-3] + 'pkl')
                        elif file[9] == 'L' and file[-5] == '2':
                            l_after_img_path = os.path.join(data_dir, sub_dir, file)
                            l_after_pkl_path = os.path.join(pkl_dir, sub_dir, file[:-3] + 'pkl')
                        elif file[9] == 'R' and file[-5] == '1':
                            r_pre_img_path = os.path.join(data_dir, sub_dir, file)
                            r_pre_pkl_path = os.path.join(pkl_dir, sub_dir, file[:-3] + 'pkl')
                        elif file[9] == 'R' and file[-5] == '2':
                            r_after_img_path = os.path.join(data_dir, sub_dir, file)
                            r_after_pkl_path = os.path.join(pkl_dir, sub_dir, file[:-3] + 'pkl')

                # add tuple in data_info
                # l_pre_img_path (str)
                # l_after_img_path (str)
                # l_img_feats (torch)
                # l_img_labels (torch)
                if l_img_index.shape[0] != 0 and l_pre_img_path != '' and l_after_img_path != '':
                    if mode == 'train':
                        data_info.append((l_patient_id, l_pre_img_path, l_after_img_path,
                                          l_pre_pkl_path, l_after_pkl_path,
                                          l_img_feats, l_img_labels))
                    else:
                        data_info.append((l_patient_id, l_pre_img_path, l_after_img_path,
                                          l_pre_pkl_path, l_after_pkl_path,
                                          l_img_feats))
                if r_img_index.shape[0] != 0 and r_pre_img_path != '' and r_after_img_path != '':
                    if mode == 'train':
                        data_info.append((r_patient_id, r_pre_img_path, r_after_img_path,
                                          r_pre_pkl_path, r_after_pkl_path,
                                          r_img_feats, r_img_labels))
                    else:
                        data_info.append((r_patient_id, r_pre_img_path, r_after_img_path,
                                          r_pre_pkl_path, r_after_pkl_path,
                                          r_img_feats))

        return data_info


if __name__ == '__main__':
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

    train_dataset = EyeDataset('../../dataset/mix_train/',
                               '../../data/train_data.pk',
                               '../../dataset/pkl_train',
                               transform=train_transform, mode='train')
    print(train_dataset.data_info[0])
    print(len(train_dataset.data_info))

    cnt = 0
    for i in range(len(train_dataset.data_info)):
        patient_id, pre_img_path, after_img_path, pre_pkl_path, after_pkl_path, img_feats, img_labels = \
            train_dataset.data_info[i]
        if pre_img_path == '' or after_img_path == '':
            cnt += 1
    print(cnt)

    test_dataset = EyeDataset('../../dataset/mix_test',
                              '../../data/test_data.pk',
                              '../../dataset/pkl_test',
                              transform=test_transform, mode='test')
    print(test_dataset.data_info[0])
    print(len(test_dataset.data_info))

    cnt = 0
    for i in range(len(test_dataset.data_info)):
        patient_id, pre_img_path, after_img_path, pre_pkl_path, after_pkl_path, img_feats = test_dataset.data_info[i]
        if pre_img_path == '' or after_img_path == '':
            cnt += 1
    print(cnt)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('train batch')
    for i, data in enumerate(train_loader):
        patient_ids, pre_img, after_img, pre_pkl, after_pkl, img_feats, img_labels = data
        pre_img, after_img, pre_pkl, after_pkl, img_feats, img_labels = pre_img.to(device), after_img.to(
            device), pre_pkl.to(device), after_pkl.to(device), img_feats.to(device), img_labels.to(device)

        # if pre_img != None:
        print(patient_ids)
        print(pre_img.shape)
        # if after_img != None:
        print(after_img.shape)
        print(img_feats.shape)
        print(pre_pkl.shape)
        print(after_pkl.shape)
        print(img_labels.shape)

        # break

    print('test batch')
    for i, data in enumerate(test_loader):
        patient_ids, pre_img, after_img, pre_pkl, after_pkl, img_feats = data
        pre_img, after_img, pre_pkl, after_pkl, img_feats = pre_img.to(device), after_img.to(device), pre_pkl.to(
            device), after_pkl.to(device), img_feats.to(device)

        print(patient_ids)
        print(pre_img.shape)
        print(after_img.shape)
        print(pre_pkl.shape)
        print(after_pkl.shape)
        print(img_feats.shape)

    # train dataset

    # ('0000-1726L',
    #  '../../dataset/mix_train/0000-1726/0000-1726L_1.jpg',
    #  '../../dataset/mix_train/0000-1726/0000-1726L_2.jpg',
    #  tensor([0.0136, 0.0244, 0.0288, 0.0075, 0.0000], dtype=torch.float64),
    #  tensor([-0.7330,  0.0000,  0.0000,  0.0000,  0.0000, -0.1084,  0.0000, -0.6558,
    #          0.0000,  0.0000,  0.0000,  0.0000], dtype=torch.float64))
    # 2131
    # 0

    #######################################

    # test

    # ('0000-0069R',
    #  '../../dataset/mix_test/0000-0069/0000-0069R_1.jpg',
    #  '../../dataset/mix_test/0000-0069/0000-0069R_2.jpg',
    #  tensor([0.0330, 0.0440, 0.0651, 0.0461, 0.1625], dtype=torch.float64))
    # 361
    # 0
