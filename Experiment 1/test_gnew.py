import os
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


def find_gt(target, ground_truth):
    min_dtw = 1000000
    cnt = -1
    for i in range(len(ground_truth)):
        tmp_gttr = ground_truth[i]
        tmp_gttr = normalize(tmp_gttr)

        dis, _ = fastdtw(target, tmp_gttr)
        # print('dtw distance', dis)
        if dis < min_dtw:
            min_dtw = dis
            cnt = i
    return normalize(ground_truth[cnt])


def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


cls_dit = {0: 'N', 1: 'S', 2: 'V', 3: 'Q', 4: 'F'}
name_dit = {0: 'Non-Ectopic Beats', 1: 'Superventrical Ectopic', 2: 'Ventricular Beats',
            3: 'Unknown', 4: 'Fusion Beats'}

# save_name = 'myself'
# ck_pth_myself = './logs/mitbithCGAN_2024_07_06_23_40_02/Model/checkpoint'
ck_pth_myself = './logs/mitbithCGAN_2024_07_07_23_05_52/Model/checkpoint'

# load ground truth
filename = './custom_mitbih_data'
# making the class labels for our dataset

gt_data = np.asarray(pd.read_csv(F"{filename}/N.csv", header=None))[:, :-1]
# random_indices = np.random.choice(gt_data.shape[0], size=100, replace=False)
# gt_data = gt_data[random_indices]

print(gt_data.shape)
num_gt = gt_data.shape[0]

ck_myself = torch.load(ck_pth_myself, map_location='cpu')
gen_net_myself = Generator(seq_len=280, channels=1, num_classes=5, latent_dim=200, data_embed_dim=10,
                           label_embed_dim=10, depth=4, num_heads=5,
                           forward_drop_rate=0.2, attn_drop_rate=0.1)

gen_net_myself.load_state_dict(ck_myself['gen_state_dict'])
gen_net_myself.to(device)
gen_net_myself.eval()

repeat_freq = 10  # 长度 = 360 * repeat_freq
save_pth = f'./ref_res/combine'
os.makedirs(save_pth, exist_ok=True)
lb = 0

print(f'label {lb}')

out_all_myself = []
ground_truth_all = []

label = torch.tensor([lb]).to(device)
noise_org = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 200))).to(device)

for j in range(repeat_freq):
    print(f"repeat time {j}")
    noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 200))).to(device)

    # noise_bias = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 200))).to(device)
    # noise = noise_org + noise_bias * 0.15

    out_myself = gen_net_myself(noise, labels=label).flatten()
    out_myself = out_myself.detach().cpu().numpy()
    out_myself = normalize(out_myself)

    out_all_myself.append(out_myself)

    # gts = find_gt(out_myself, gt_data)
    random_number = random.randint(0, len(gt_data))
    gts = gt_data[random_number]

    ground_truth_all.append(gts)

out_all_myself = np.concatenate(out_all_myself, axis=-1)
ground_truth_all = np.concatenate(ground_truth_all, axis=-1)

print(out_all_myself.shape)

# metrics
out_myself = out_all_myself
tmp_ground_truth = ground_truth_all
print('myself metrics')
dis, _ = fastdtw(tmp_ground_truth, out_myself)
print('dtw distance', dis)
rmse = np.sqrt(np.mean((tmp_ground_truth - out_myself) ** 2))
print('RMSE', rmse)
Q = np.moveaxis(np.asarray([[i, out_myself[i]] for i in range(len(out_myself))]), 0, -1)
P = np.moveaxis(np.asarray([[i, tmp_ground_truth[i]] for i in range(len(tmp_ground_truth))]), 0, -1)
frechet_distance = distance.directed_hausdorff(Q, P)[0]
print('frechet distance', frechet_distance)
# mmd_value = mmd_rbf(tmp_ground_truth, out_myself)
# print("MMD", mmd_value)

fig, axs = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle(f"Comparison of Generated ECG Signal and Groundtruth")  # 大标题
axs[0].plot(out_all_myself)
axs[0].set_title('Generated ECG Signal')
axs[0].set_xlabel('Sampling points')
axs[0].set_ylabel('Voltage')

axs[1].plot(ground_truth_all)
axs[1].set_title('Groundtruth ECG Signal')
axs[1].set_xlabel('Sampling points')
axs[1].set_ylabel('Voltage')

plt.savefig(f'./{save_pth}/{lb}.png')
plt.show()
