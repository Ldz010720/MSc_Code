# load mitbih dataset

import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

cls_dit = {'Non-Ectopic Beats': 0, 'Superventrical Ectopic': 1, 'Ventricular Beats': 2,
           'Unknown': 3, 'Fusion Beats': 4}

AAMI_MIT = {'N': 'Nfe/jnBLR',  # 将19类信号分为五大类
            'S': 'SAJa',
            'V': 'VEr',
            'F': 'F',
            'Q': 'Q?'}


def standardize(matrix):
    # 计算整个矩阵的均值和标准差
    mean = np.mean(matrix)
    std = np.std(matrix)
    print('mean, std', mean, std)
    # 进行标准化
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix


class mitbih_train_custom(Dataset):
    def __init__(self, filedir='./custom_mitbih_data', n_samples=20000, oneD=False):

        # the class labels for our dataset
        data_0 = pd.read_csv(f"{filedir}/N.csv", header=None)
        data_1 = pd.read_csv(f"{filedir}/S.csv", header=None)
        data_2 = pd.read_csv(f"{filedir}/V.csv", header=None)
        data_3 = pd.read_csv(f"{filedir}/Q.csv", header=None)
        data_4 = pd.read_csv(f"{filedir}/F.csv", header=None)

        print(data_0.shape, data_1.shape, data_2.shape, data_3.shape, data_4.shape)

        data_0_resample = resample(data_0, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_1_resample = resample(data_1, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_2_resample = resample(data_2, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_3_resample = resample(data_3, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_4_resample = resample(data_4, n_samples=n_samples,
                                   random_state=123, replace=True)
        print('after resample')
        print(data_0_resample.shape, data_1_resample.shape, data_2_resample.shape, data_3_resample.shape,
              data_4_resample.shape)

        train_dataset = pd.concat((data_0_resample, data_1_resample,
                                   data_2_resample, data_3_resample, data_4_resample))

        self.X_train = train_dataset.iloc[:, :-1].values  # b,d
        if oneD:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        else:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, 1, self.X_train.shape[1])
        self.y_train = train_dataset[280].values

        self.X_train = standardize(self.X_train)

        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print(
            f'The dataset including {len(data_0_resample)} class 0, {len(data_1_resample)} class 1, {len(data_2_resample)} class 2, {len(data_3_resample)} class 3, {len(data_4_resample)} class 4')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


if __name__ == '__main__':
    kkk = mitbih_train_custom()
