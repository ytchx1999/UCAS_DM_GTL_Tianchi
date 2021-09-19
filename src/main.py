import numpy as np
import pandas as pd
import torch
import pickle


def main():
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


# main
if __name__ == '__main__':
    main()
