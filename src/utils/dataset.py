import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import pickle


# data_dir = ../../dataset/mix_train/
# 0000-0000/0000-0000L_1.jpg
class EyeDataset(Dataset):
    def __init__(self, data_dir, csv_dir, transform=None):
        super(EyeDataset, self).__init__()
        self.data_dir = data_dir
        self.csv_dir = csv_dir
        self.data_info = self.get_data(self.data_dir, self.csv_dir)
        self.transform = transform

    def __getitem__(self, item):
        pre_img_path, after_img_path, img_feats, img_labels = self.data_info[item]
        pre_img = Image.open(pre_img_path).convert('RGB')
        after_img = Image.open(after_img_path).convert('RGB')
        if self.transform != None:
            pre_img = self.transform(pre_img)
            after_img = self.transform(after_img)
        return pre_img, after_img, img_feats, img_labels

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_data(data_dir, csv_dir):
        with open(csv_dir, 'rb') as f:
            id_index, feats, labels = pickle.load(f)
        data_info = []
        for _, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                if sub_dir[:9] in id_index:
                    img_index = np.argwhere(id_index == sub_dir[:9])
                else:
                    continue
                patient_id = id_index[img_index]
                img_feats = feats[img_index]
                img_labels = labels[img_index]

                # pre, after img_path
                l_pre_img_path = ''
                l_after_img_path = ''
                r_pre_img_path = ''
                r_after_img_path = ''
                for _, _, files in os.walk(os.path.join(data_dir, sub_dir)):
                    for file in files:
                        if file[9] == 'L' and file[-5] == '1':
                            l_pre_img_path += os.path.join(data_dir, sub_dir, file)
                        elif file[9] == 'L' and file[-5] == '2':
                            l_after_img_path += os.path.join(data_dir, sub_dir, file)
                        elif file[9] == 'R' and file[-5] == '1':
                            r_pre_img_path += os.path.join(data_dir, sub_dir, file)
                        elif file[9] == 'R' and file[-5] == '2':
                            r_after_img_path += os.path.join(data_dir, sub_dir, file)

                        if file[9] == 'L':
                            data_info.append((l_pre_img_path, l_after_img_path, img_feats, img_labels))
                        elif file[9] == 'R':
                            data_info.append((r_pre_img_path, r_after_img_path, img_feats, img_labels))
                # img_names = os.listdir(os.path.join(data_dir, sub_dir))
                # for img_name in img_names:
                #     img_path = os.path.join(data_dir, sub_dir, img_name)
                #     label = label_mp[sub_dir]
                #     data_info.append((img_path, int(label)))
        return data_info


if __name__ == '__main__':
    train_dataset = EyeDataset('../../dataset/mix_train/', '../../data/train_data.pk', transform=None)
    print(train_dataset.data_info)
