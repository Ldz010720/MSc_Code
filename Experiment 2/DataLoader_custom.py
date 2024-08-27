# load mitbih dataset
import glob
import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample
import matplotlib.pyplot as plt

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

def standardize(matrix):
    # 计算整个矩阵的均值和标准差
    mean = np.nanmean(matrix)
    std = np.nanstd(matrix)
    print('mean, std', mean, std)
    # 进行标准化
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix

class mitbih_train_custom(Dataset):
    def __init__(self, filedir='./my_data', n_samples=20000, oneD=False, args=None):
        data_list = glob.glob(f"{filedir}/**/*.npy", recursive=True)
        data = []

        channel_id = args.gen_channel

        for pth in data_list:
            print(pth)
            data.append(np.load(pth))

        data = np.concatenate(data, axis=0)
        print(data.shape)

        # 去nan
        rows_with_nan = np.isnan(data).any(axis=(1, 2))
        data = data[~rows_with_nan]
        print(data.shape)

        self.X_train = data
        print(self.X_train.shape)

        self.X_train = self.X_train[:, channel_id, :]
        print(self.X_train.shape)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, 1, self.X_train.shape[1]))
        self.X_train = standardize(self.X_train)

        print(f'X_train shape is {self.X_train.shape}')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], 0


if __name__ == '__main__':
    kkk = mitbih_train_custom()

    # for i in range(20):
    #     ig, _ = kkk[i]
    #
    #     plt.figure()
    #     plt.plot(ig.flatten())
    #     plt.show()
