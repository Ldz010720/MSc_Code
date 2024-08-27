import wfdb  # pip install wfdb
import glob
import numpy as np
import cv2
import neurokit2  # pip install neurokit2
import sleepecg  # pip install sleepecg
import wfdb.processing
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import signal, interpolate


# # OPTION 1: very fast, good performance, large user-base
# _, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=360)
# rpeaks = results["ECG_R_Peaks"]
# # OPTION 2: blazingly fast, solid performance, relatively uncommon
# rpeaks = sleepecg.detect_heartbeats(ecg_signal, fs=360)
#
# # OPTION 3: excellent performance, but slower, from MIT researchers
# rpeaks = wfdb.processing.xqrs_detect(ecg_signal, fs=360, verbose=False)



def normalize_01(row):
    min_val = np.min(row)
    max_val = np.max(row)
    return (row - min_val) / (max_val - min_val)


# file_nm = '01'
file_nm = '02'
root = rf'E:\chb_datas\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords\{file_nm}'
save_name = f'data{file_nm}'

file_list = glob.glob(f"{root}/**/*.hea", recursive=True)
file_list = [x.rstrip('.hea') for x in file_list]
print(len(file_list))
# file_list = file_list[:3]

all_data = []

for pth in file_list:
    print(pth)
    try:
        record = wfdb.rdrecord(pth)
        ventricular_signal = record.p_signal
    except:
        continue

    ecg_signal = np.moveaxis(ventricular_signal, 0, -1)
    # ecg_signal = ecg_signal[:, :2000]
    print(ecg_signal.shape)

    tmp_sg = ecg_signal[-2]
    lth = 560

    _, results = neurokit2.ecg_peaks(tmp_sg, sampling_rate=lth)
    rpeaks = results["ECG_R_Peaks"]
    print(rpeaks)

    plt.figure()
    plt.title(pth)
    plt.plot(tmp_sg)
    plt.plot(rpeaks, tmp_sg[rpeaks], "x")
    plt.show()

    for xx in rpeaks:
        if xx - 140 < 0 or xx + 140 > ecg_signal.shape[1]:
            continue
        tmp = ecg_signal[:, xx - 140:xx + 140]
        tmp = np.apply_along_axis(normalize_01, 1, tmp)

        # for i in range(12):
        #     plt.figure()
        #     plt.title(f"{i}")
        #     plt.plot(tmp[i])
        #     # plt.plot(rpeaks, tmp[i][xx], "x")
        #     plt.show()

        tmp = np.expand_dims(tmp, axis=0)
        all_data.append(tmp)

        break
    break

# all_data = np.concatenate(all_data, axis=0)
# print(all_data.shape)
# np.save(f'./my_data/{file_nm}.npy', all_data)
