import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn.functional as F


def preprocess_csv():
    '''
    train_id_index: dict, {'patient ID': 0, ...}
    train_feats: torch_tensor, L2 normalization
    train_labels_dict: dict, {'preCST': torch.tensor(...), ...}

    test_id_index: dict, {'patient ID': 0, ...}
    test_feats: torch_tensor, L2 normalization

    :return: None, saved in ../../data/train_data.pk, ../../data/test_data.pk
    '''
    ########################
    # train
    ########################

    train_data = pd.read_csv('../../data/TrainingAnnotation.csv')
    print(train_data.shape)
    # 去除包含nan的行
    train_data = train_data.dropna()
    print(train_data.shape)

    # patient ID as index (dict)
    train_id_index = train_data['patient ID'].values
    train_id_index = {key: i for i, key in enumerate(train_id_index)}
    print('train id:', train_id_index)

    # train features (torch)
    train_feats = train_data[['gender', 'age', 'diagnosis', 'preVA', 'anti-VEGF']].values
    train_feats = torch.from_numpy(train_feats)
    # L2 Normalize
    train_feats = F.normalize(train_feats, p=2, dim=0)
    print('train feats:', train_feats)

    # train labels (torch)
    # train_labels = train_data[['preCST', 'preIRF', 'preSRF',
    #                            'prePED', 'preHRF', 'VA',
    #                            'continue injection', 'CST', 'IRF',
    #                            'SRF', 'PED', 'HRF']].values

    # train labels (dict_torch)
    train_labels_dict = {
        'preCST': torch.from_numpy(train_data[['preCST']].values),
        'preIRF': torch.from_numpy(train_data[['preIRF']].values).long(),
        'preSRF': torch.from_numpy(train_data[['preSRF']].values).long(),
        'prePED': torch.from_numpy(train_data[['prePED']].values).long(),
        'preHRF': torch.from_numpy(train_data[['preHRF']].values).long(),
        'VA': torch.from_numpy(train_data[['VA']].values),
        'continue injection': torch.from_numpy(train_data[['continue injection']].values).long(),
        'CST': torch.from_numpy(train_data[['CST']].values),
        'IRF': torch.from_numpy(train_data[['IRF']].values).long(),
        'SRF': torch.from_numpy(train_data[['SRF']].values).long(),
        'PED': torch.from_numpy(train_data[['PED']].values).long(),
        'HRF': torch.from_numpy(train_data[['HRF']].values).long()
    }
    # train_labels = torch.from_numpy(train_labels)
    print('train labels dict:', train_labels_dict)

    # save variables in '../data/train_data.pk'
    with open('../../data/train_data.pk', 'wb') as f:
        pickle.dump((train_id_index, train_feats, train_labels_dict), f)

    ########################
    # test
    ########################

    test_data = pd.read_csv('../../data/PreliminaryValidationSet_Info.csv')
    print(test_data.shape)

    # patient ID as index (dict)
    test_id_index = test_data['patient ID'].values
    test_id_index = {key: i for i, key in enumerate(test_id_index)}
    print('test id:', test_id_index)

    # test features (torch)
    test_feats = test_data[['gender', 'age', 'diagnosis', 'preVA', 'anti-VEGF']].values
    test_feats = torch.from_numpy(test_feats)
    test_feats = F.normalize(test_feats, p=2, dim=0)
    print('test feats:', test_feats)

    # save variables in '../data/test_data.pk'
    with open('../../data/test_data.pk', 'wb') as f:
        pickle.dump((test_id_index, test_feats), f)


if __name__ == '__main__':
    preprocess_csv()