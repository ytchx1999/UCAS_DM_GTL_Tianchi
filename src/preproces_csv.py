import numpy as np
import pandas as pd
import torch
import pickle

########################
# train
########################

train_data = pd.read_csv('../data/TrainingAnnotation.csv')
print(train_data.shape)
# 去除包含nan的行
train_data = train_data.dropna()
print(train_data.shape)

# patient ID as index (numpy)
train_id_index = train_data['patient ID'].values

# train features (torch)
train_feats = train_data[['gender', 'age', 'diagnosis', 'preVA', 'anti-VEGF']].values
train_feats = torch.from_numpy(train_feats)

# train labels (torch)
train_labels = train_data[['preCST', 'preIRF', 'preSRF',
                           'prePED', 'preHRF', 'VA',
                           'continue injection', 'CST', 'IRF',
                           'SRF', 'PED', 'HRF']].values
train_labels = torch.from_numpy(train_labels)

# save variables in '../data/train_data.pk'
with open('../data/train_data.pk', 'wb') as f:
    pickle.dump((train_id_index, train_feats, train_labels), f)

########################
# test
########################

test_data = pd.read_csv('../data/PreliminaryValidationSet_Info.csv')
print(test_data.shape)

# patient ID as index (numpy)
test_id_index = test_data['patient ID'].values

# test features (torch)
test_feats = test_data[['gender', 'age', 'diagnosis', 'preVA', 'anti-VEGF']].values
test_feats = torch.from_numpy(test_feats)

# save variables in '../data/test_data.pk'
with open('../data/test_data.pk', 'wb') as f:
    pickle.dump((test_id_index, test_feats), f)