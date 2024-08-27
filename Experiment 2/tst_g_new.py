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
import neurokit2

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


def cal_metrics(out_myself, gt):
    dis, _ = fastdtw(gt, out_myself)
    print('dtw distance', dis)
    rmse = np.sqrt(np.mean((gt - out_myself) ** 2))
    print('RMSE', rmse)
    Q = np.moveaxis(np.asarray([[i, out_myself[i]] for i in range(len(out_myself))]), 0, -1)
    P = np.moveaxis(np.asarray([[i, gt[i]] for i in range(len(gt))]), 0, -1)
    frechet_distance = distance.directed_hausdorff(Q, P)[0]
    print('frechet distance', frechet_distance)
    # MMD计算很慢
    # mmd_value = mmd_rbf(gt, out_myself)
    # print("MMD", mmd_value)


def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


y_labels = ['Ⅰ', 'Ⅱ', 'Ⅲ', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

gtttt = np.load('./my_data/01.npy')
rows_with_nan = np.isnan(gtttt).any(axis=(1, 2))
gtttt = gtttt[~rows_with_nan]
print(gtttt.shape)

random_indices = np.random.choice(gtttt.shape[0], size=100, replace=False)
gtttt = gtttt[random_indices]
print(gtttt.shape)

# channel = 11
# ground_truth = ground_truth_all[:, channel, :]
# num_gt = ground_truth.shape[0]

all_model = []
for channel in range(12):
    ck_pth_myself = f'./logs/mitbithCGAN_channel{channel}/Model/checkpoint'
    ck_myself = torch.load(ck_pth_myself, map_location='cpu')
    gen_net_myself = Generator(seq_len=280, channels=1, num_classes=1, latent_dim=200, data_embed_dim=10,
                               label_embed_dim=10, depth=4, num_heads=5,
                               forward_drop_rate=0.2, attn_drop_rate=0.1)
    gen_net_myself.load_state_dict(ck_myself['gen_state_dict'])
    gen_net_myself.to(device)
    gen_net_myself.eval()

    all_model.append(gen_net_myself)

hz = 500  # 就是500
repeat_freq = 5  # 长度 = 360 * repeat_freq
save_pth = f'./ref_res'
os.makedirs(save_pth, exist_ok=True)

label = torch.tensor([0]).to(device)
noise_org = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 200))).to(device)

fig, axs = plt.subplots(12, 1, figsize=(40, 20))  # (宽 高)
fig.subplots_adjust(hspace=0.75)  # 这里增加子图之间的间距
# fig.suptitle(f"Generated ECG Signal", fontsize=50)

fig_gt, axs_gt = plt.subplots(12, 1, figsize=(40, 20))  # (宽 高)
fig_gt.subplots_adjust(hspace=0.75)  # 这里增加子图之间的间距
# fig_gt.suptitle(f"Ground truth ECG Signal", fontsize=50)


all_channel_out = []
all_channel_gt = []

# 更新后的绘图循环代码部分
# 更新后的绘图循环代码部分
for cn in range(12):
    print(f"start channel {cn}")
    out_all_myself = []
    ground_truth_all = []
    for j in range(repeat_freq):
        print(f"repeat time {j}")

        noise_bias = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 200))).to(device)
        noise = noise_org + noise_bias * 0.1

        out_myself = all_model[cn](noise, labels=label).flatten()
        out_myself = out_myself.detach().cpu().numpy()
        out_myself = normalize(out_myself)

        out_all_myself.append(out_myself)

        # gt
        tmp_gt = gtttt[:, cn, :]
        grt = find_gt(out_myself, tmp_gt)
        ground_truth_all.append(grt)

    out_all_myself = np.concatenate(out_all_myself, axis=-1)
    ground_truth_all = np.concatenate(ground_truth_all, axis=-1)

    all_channel_out.append(out_all_myself)
    all_channel_gt.append(ground_truth_all)

    x = np.linspace(0, len(out_all_myself.flatten()) / hz, len(out_all_myself.flatten()))
    y = out_all_myself.flatten()

    axs[cn].set_yticks([])
    axs[cn].plot(x, y)
    axs[cn].set_xlim(x[0] - 0.1, x[-1] + 0.1)

    # 只在最后一个通道显示横坐标标题
    if cn == 11:
        axs[cn].set_xlabel('Time (s)', fontsize=40)
    axs[cn].set_ylabel(y_labels[cn], fontsize=30)
    axs[cn].tick_params(axis='x', labelsize=20)

    y_gt = ground_truth_all.flatten()
    axs_gt[cn].set_yticks([])
    axs_gt[cn].plot(x, y_gt)
    axs_gt[cn].set_xlim(x[0] - 0.1, x[-1] + 0.1)

    if cn == 11:
        axs_gt[cn].set_xlabel('Time (s)', fontsize=40)
    axs_gt[cn].set_ylabel(y_labels[cn], fontsize=30)
    axs_gt[cn].tick_params(axis='x', labelsize=20)

    # 计算指标
    print(f'Metrics channel {cn}')
    cal_metrics(y, y_gt)
    print()


# plt.show()
fig.subplots_adjust(top=0.92)
fig_gt.subplots_adjust(top=0.92)
fig.savefig(f'./{save_pth}/Generated ECG signal.png')
fig_gt.savefig(f'./{save_pth}/Ground truth.png')
print()

# 寻找peak
save_pth = f'./ref_res/peaks'
os.makedirs(save_pth, exist_ok=True)

print('Peek')
for i in range(12):
    print('channel', i)
    tmp_out = all_channel_out[i]
    _, results = neurokit2.ecg_peaks(tmp_out, sampling_rate=240)
    rpeaks = results["ECG_R_Peaks"]
    print('out', rpeaks)
    plt.figure(figsize=(16, 8))
    plt.plot(tmp_out)
    plt.plot(rpeaks, tmp_out[rpeaks], "x")
    plt.yticks([])
    os.makedirs(f'./{save_pth}/out/', exist_ok=True)
    plt.savefig(f'./{save_pth}/out/out_channel{i}.png')

    tmp_out = all_channel_gt[i]
    _, results = neurokit2.ecg_peaks(tmp_out, sampling_rate=560)
    rpeaks = results["ECG_R_Peaks"]
    print('out', rpeaks)
    plt.figure(figsize=(16, 8))
    plt.plot(tmp_out)
    plt.plot(rpeaks, tmp_out[rpeaks], "x")
    plt.yticks([])
    os.makedirs(f'./{save_pth}/gt/', exist_ok=True)
    plt.savefig(f'./{save_pth}/gt/out_channel{i}.png')

    print()
