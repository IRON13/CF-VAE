import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


def remove_missing(full_data):
    full_size = full_data.shape[0]
    for data in [full_data]:
        for i in full_data:
            data[i].replace('nan', np.nan, inplace=True)
        data.dropna(inplace=True)
    real_size = full_data.shape[0]


def replace_categorical(full_data):
    cat_data = full_data.select_dtypes(include=['object']).copy()
    other_data = full_data.select_dtypes(include=['int']).copy()

    cat_data[["Workclass", "Country", "Martial Status", "Occupation", "Relationship", "Race", "Sex"]] = cat_data[["Workclass", "Country", "Martial Status", "Occupation", "Relationship", "Race", "Sex"]].apply(LabelEncoder().fit_transform)

    return pd.concat([other_data, cat_data], axis=1)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def load_data():
    columns = ["Age", "Workclass", "Education-Num", "Martial Status", \
               "Occupation", "Relationship", "Race", "Sex",
               "Hours per week", "Country", "Target"]

    types = {0: int, 1: str, 2: int, 3: str, 4: str, 5: str, 6: str, 7: str, 8: int, 9: str, 10: int}

    df = pd.read_csv(
        "Dataset/adult.csv",
        names=columns,
        sep=r'\s*,\s*',
        engine='python', skiprows=1,
        na_values="?",
        dtype=types)

    remove_missing(df)
    df = replace_categorical(df)
    df = df[["Sex","Race","Age","Country","Education-Num","Workclass","Occupation","Hours per week","Martial Status","Relationship","Target"]]
    df = normalization(df)

    df_train, df_test = train_test_split(df, test_size=0.3)

    target_feature_1 = 'Target'
    train_X = df_train.drop([target_feature_1], axis=1)
    train_Y = df_train[target_feature_1]

    test_X = df_test.drop([target_feature_1], axis=1)
    test_Y = df_test[target_feature_1]

    test_X_cf = df[(df['Sex'] == 0) & (df['Target']==0)]
    test_X_cf = test_X_cf.sample(n=10000, frac=None, replace=False, weights=None, random_state=None, axis=None)
    test_Y_cf = test_X_cf[target_feature_1]
    test_X_cf = test_X_cf.drop([target_feature_1], axis=1)


    for i in range(test_X_cf.shape[0]):
        test_X_cf.iat[i, 0] = 1-test_X_cf.iat[i, 0]

    return torch.from_numpy(train_X.reset_index().values[:, 1:]), \
           torch.from_numpy(train_Y.reset_index().values[:, 1:].reshape(-1)), \
           torch.from_numpy(test_X.reset_index().values[:, 1:]), \
           torch.from_numpy(test_X_cf.reset_index().values[:, 1:]), \
           torch.from_numpy(test_Y.reset_index().values[:, 1:].reshape(-1)), \
           torch.from_numpy(test_Y_cf.reset_index().values[:, 1:].reshape(-1))
