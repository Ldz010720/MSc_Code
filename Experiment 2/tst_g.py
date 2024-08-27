import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
from TransCGAN_model import *
import matplotlib.pyplot as plt
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial import distance
from mmd import mmd_rbf
from scipy.signal import medfilt
from torch.nn import functional as F
from scipy.signal import butter, filtfilt
import random

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def standardize(matrix):
    # 计算整个矩阵的均值和标准差
    mean = np.nanmean(matrix)
    std = np.nanstd(matrix)
    print('mean, std', mean, std)
    # 进行标准化
    standardized_matrix = (matrix - mean) / std
    return standardized_matrix

def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


ground_truth_all = np.load('./my_data/01.npy')

channel = 11
ground_truth = ground_truth_all[:, channel, :]
# print(ground_truth)
# ground_truth = standardize(ground_truth)
num_gt = ground_truth.shape[0]

ck_pth_myself = f'./logs/mitbithCGAN_channel{channel}/Model/checkpoint'
ck_myself = torch.load(ck_pth_myself, map_location='cpu')
gen_net_myself = Generator(seq_len=280, channels=1, num_classes=1, latent_dim=200, data_embed_dim=10,
                           label_embed_dim=10, depth=4, num_heads=5,
                           forward_drop_rate=0.2, attn_drop_rate=0.1)

gen_net_myself.load_state_dict(ck_myself['gen_state_dict'])
gen_net_myself.to(device)
gen_net_myself.eval()

repeat_freq = 6  # 长度 = 360 * repeat_freq
save_pth = f'./ref_res/combine'
os.makedirs(save_pth, exist_ok=True)

out_all_myself = []
ground_truth_all = []

label = torch.tensor([0]).to(device)
noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 200))).to(device)

for j in range(repeat_freq):
    print(f"repeat time {j}")

    noise_bias = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 200))).to(device)
    noise = noise + noise_bias * 0.1

    out_myself = gen_net_myself(noise, labels=label).flatten()
    out_myself = out_myself.detach().cpu().numpy()
    # print(out_myself)
    out_myself = normalize(out_myself)
    out_all_myself.append(out_myself)

    k = random.randint(0, num_gt)
    ground_truth_all.append(normalize(ground_truth[k]))

out_all_myself = np.concatenate(out_all_myself, axis=-1)
ground_truth_all = np.concatenate(ground_truth_all, axis=-1)

print(out_all_myself.shape, ground_truth_all.shape)

# plt.plot(out_all_myself.flatten())
# plt.show()

fig, axs = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle(f"channel {channel}")
axs[0].plot(out_all_myself.flatten())
axs[0].set_title('generated myself')
axs[1].plot(ground_truth_all.flatten())
axs[1].set_title('groundtruth')
# plt.savefig(f'./{save_pth}/{lb}.png')
plt.show()

print()
