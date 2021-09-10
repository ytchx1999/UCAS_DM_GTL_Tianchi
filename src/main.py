import numpy as np
import pandas as pd
import torch
import pickle


def main():
    # load train data
    with open('../data/train_data.pk', 'rb') as f:
        train_id_index, train_feats, train_labels = pickle.load(f)
    print(
        'train_id_index.shape=', train_id_index.shape,
        'train_feats.shape=', train_feats.shape,
        'train_labels.shape=', train_labels.shape
    )

    # load test data
    with open('../data/test_data.pk', 'rb') as f:
        test_id_index, test_feats = pickle.load(f)
    print(
        'test_id_index.shape=', test_id_index.shape,
        'test_feats.shape=', test_feats.shape
    )


# main
if __name__ == '__main__':
    main()
