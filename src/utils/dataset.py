import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import pickle


# data_dir = ../../dataset/mix_train/
# 0000-0000/0000-0000L_1.jpg
class EyeDataset(Dataset):
    def __init__(self, data_dir, csv_dir, transform=None, mode='train'):
        super(EyeDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.mode = mode
        self.data_info = self.get_data(self.data_dir, self.csv_dir, self.mode)
        self.transform = transform

    def __getitem__(self, item):
        '''
        :param item: index
        :return: if train, return pre_img, after_img, img_feats, img_labels
                 else, return pre_img, after_img, img_feats (no labels)
        '''
        if self.mode == 'train':
            patient_id, pre_img_path, after_img_path, img_feats, img_labels = self.data_info[item]
        else:
            patient_id, pre_img_path, after_img_path, img_feats = self.data_info[item]

        if pre_img_path != '':
            pre_img = Image.open(pre_img_path).convert('RGB')
        else:
            pre_img = None
        if after_img_path != '':
            after_img = Image.open(after_img_path).convert('RGB')
        else:
            after_img = None

        if self.transform != None:
            if pre_img_path != '':
                pre_img = self.transform(pre_img)
            if after_img_path != '':
                after_img = self.transform(after_img)

        if self.mode != 'train':
            return patient_id, pre_img, after_img, img_feats
        return patient_id, pre_img, after_img, img_feats, img_labels

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_data(data_dir, csv_dir, mode):
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

                for _, _, files in os.walk(os.path.join(data_dir, sub_dir)):
                    for file in files:
                        # print(file)
                        if file[9] == 'L' and file[-5] == '1':
                            l_pre_img_path = os.path.join(data_dir, sub_dir, file)
                        elif file[9] == 'L' and file[-5] == '2':
                            l_after_img_path = os.path.join(data_dir, sub_dir, file)
                        elif file[9] == 'R' and file[-5] == '1':
                            r_pre_img_path = os.path.join(data_dir, sub_dir, file)
                        elif file[9] == 'R' and file[-5] == '2':
                            r_after_img_path = os.path.join(data_dir, sub_dir, file)

                # add tuple in data_info
                # l_pre_img_path (str)
                # l_after_img_path (str)
                # l_img_feats (torch)
                # l_img_labels (torch)
                if l_img_index.shape[0] != 0 and l_pre_img_path != '' and l_after_img_path != '':
                    if mode == 'train':
                        data_info.append((l_patient_id, l_pre_img_path, l_after_img_path, l_img_feats, l_img_labels))
                    else:
                        data_info.append((l_patient_id, l_pre_img_path, l_after_img_path, l_img_feats))
                if r_img_index.shape[0] != 0 and r_pre_img_path != '' and r_after_img_path != '':
                    if mode == 'train':
                        data_info.append((r_patient_id, r_pre_img_path, r_after_img_path, r_img_feats, r_img_labels))
                    else:
                        data_info.append((r_patient_id, r_pre_img_path, r_after_img_path, r_img_feats))

        return data_info


if __name__ == '__main__':
    train_dataset = EyeDataset('../../dataset/mix_train/', '../../data/train_data.pk', transform=None, mode='train')
    print(train_dataset.data_info[0])
    print(len(train_dataset.data_info))

    cnt = 0
    for i in range(len(train_dataset.data_info)):
        patient_id, pre_img_path, after_img_path, img_feats, img_labels = train_dataset.data_info[i]
        if pre_img_path == '' or after_img_path == '':
            cnt += 1
    print(cnt)

    test_dataset = EyeDataset('../../dataset/mix_test', '../../data/test_data.pk', transform=None, mode='test')
    print(test_dataset.data_info[0])
    print(len(test_dataset.data_info))

    cnt = 0
    for i in range(len(test_dataset.data_info)):
        patient_id, pre_img_path, after_img_path, img_feats = test_dataset.data_info[i]
        if pre_img_path == '' or after_img_path == '':
            cnt += 1
    print(cnt)

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
