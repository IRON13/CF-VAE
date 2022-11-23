import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def load_data():
    df = pd.read_csv('Dataset/law.csv')
    df['LSAT'] = normalization(df['LSAT'])
    df['UGPA'] = normalization(df['UGPA'])
    #print(df)

    df_train, df_test = train_test_split(df, test_size = 0.3)

    target_feature_1 = 'ZFYA'

    train_X = df_train.drop([target_feature_1], axis=1)
    train_Y = df_train[target_feature_1]

    test_X = df_test.drop([target_feature_1], axis=1)
    test_Y = df_test[target_feature_1]

    test_X_cf = df_test.drop([target_feature_1], axis=1)

    for i in range(test_X_cf.shape[0]):
        if test_X_cf.iat[i,4] + test_X_cf.iat[i,9] != 0:
            test_X_cf.iat[i,4] = 1 - test_X_cf.iat[i,4]
            test_X_cf.iat[i,9] = 1 - test_X_cf.iat[i,9]

    return torch.from_numpy(train_X.reset_index().values[:, 1:]), \
           torch.from_numpy(train_Y.reset_index().values[:, 1:].reshape(-1)), \
           torch.from_numpy(test_X.reset_index().values[:, 1:]), \
           torch.from_numpy(test_X_cf.reset_index().values[:, 1:]), \
           torch.from_numpy(test_Y.reset_index().values[:, 1:].reshape(-1))