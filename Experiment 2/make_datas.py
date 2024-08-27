import os

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path
import wfdb
import glob
from scipy import interpolate
import cv2


def split(main_array, sub_length=120):
    # 可以完整采样的次数
    total_samples = main_array.shape[1] // sub_length

    # 存储所有采样的子数组
    sub_arrays = []

    for i in range(total_samples):
        start_index = i * sub_length
        end_index = start_index + sub_length
        sub_array = main_array[:, start_index:end_index]
        sub_arrays.append(np.expand_dims(sub_array, 0))
    return sub_arrays


file_nm = '01'
root = rf'E:\chb_datas\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords\{file_nm}'
save_name = f'data{file_nm}'

file_list = glob.glob(f"{root}/**/*.hea", recursive=True)
file_list = [x.rstrip('.hea') for x in file_list]
print(len(file_list))

all_list = []
for pth in file_list:
    try:
        print(pth)
        record = wfdb.rdrecord(pth)
        ventricular_signal = record.p_signal

        ventricular_signal = np.moveaxis(ventricular_signal, 0, -1)
        print(ventricular_signal.shape)

        ventricular_signal = cv2.resize(ventricular_signal, (2000, 12))
        print(ventricular_signal.shape)

        # for i in range(12):
        #     plt.figure()
        #     nm = pth.split('\\')[-1]
        #     plt.title(f"{nm}_{i}")
        #     plt.plot(ventricular_signal[i,])
        #     plt.show()

        comments = record.comments
        print(comments)

        print(ventricular_signal.shape)
        print(len(ventricular_signal))

        ventricular_signal_list = split(ventricular_signal, sub_length=280)
        all_list.extend(ventricular_signal_list)

    except:
        print('error')
        continue

    # break

all_list = np.concatenate(all_list, axis=0)

print(all_list.shape)
os.makedirs('./my_data/', exist_ok=True)
np.save(f'./my_data/{save_name}.npy', all_list)
