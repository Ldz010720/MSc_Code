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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# save_name = 'opensource'
ck_pth = './mitbih_checkpoint'
# save_name = 'myself'
ck_pth_myself = './logs/mitbithCGAN_2024_06_25_23_24_29/Model/checkpoint'

# load ground truth
filename = './mitbih_train.csv'
data_train = pd.read_csv(filename, header=None)
# making the class labels for our dataset
ground_truth = []
for i in range(5):
    data = np.asarray(data_train[data_train[187] == i])[0][:-1]
    ground_truth.append(data)

ck = torch.load(ck_pth, map_location='cpu')
# print(ck)
gen_net = Generator(seq_len=187, channels=1, num_classes=5, latent_dim=100, data_embed_dim=10,
                    label_embed_dim=10, depth=3, num_heads=5,
                    forward_drop_rate=0.5, attn_drop_rate=0.5)

gen_net.load_state_dict(ck['gen_state_dict'])
gen_net.to(device)
gen_net.eval()

noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (1, 100))).to(device)
label = torch.tensor([0]).to(device)
out = gen_net(noise, labels=label).flatten()
out = out.detach().cpu().numpy()

