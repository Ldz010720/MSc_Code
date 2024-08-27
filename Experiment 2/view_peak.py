import wfdb  # pip install wfdb
import glob
import numpy as np
import cv2
import neurokit2  # pip install neurokit2
import sleepecg  # pip install sleepecg
import wfdb.processing
import matplotlib.pyplot as plt

file_nm = '01'
root = rf'E:\chb_datas\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords\{file_nm}'
save_name = f'data{file_nm}'

file_list = glob.glob(f"{root}/**/*.hea", recursive=True)
file_list = [x.rstrip('.hea') for x in file_list]
print(len(file_list))
file_list = file_list[:3]

pth = file_list[0]

record = wfdb.rdrecord(pth)
ventricular_signal = record.p_signal

ecg_signal = np.moveaxis(ventricular_signal, 0, -1)
ecg_signal = ecg_signal[:, :1000]
print(ecg_signal.shape)
# ecg_signal = cv2.resize(ventricular_signal, (3000, 12))
# print(ecg_signal.shape)
ecg_signal_org = ecg_signal
ecg_signal = ecg_signal[-1]
print(ecg_signal.shape)

# plt.figure()
# plt.plot(ecg_signal)
# plt.show()

# # OPTION 1: very fast, good performance, large user-base
_, results = neurokit2.ecg_peaks(ecg_signal, sampling_rate=360)
rpeaks = results["ECG_R_Peaks"]
print(rpeaks)
#
# # OPTION 2: blazingly fast, solid performance, relatively uncommon
# rpeaks = sleepecg.detect_heartbeats(ecg_signal, fs=360)
#
# # OPTION 3: excellent performance, but slower, from MIT researchers
# rpeaks = wfdb.processing.xqrs_detect(ecg_signal, fs=360, verbose=False)

# 可视化
# less fancy:
plt.figure()
plt.plot(ecg_signal)
plt.plot(rpeaks, ecg_signal[rpeaks], "x")
plt.show()

for i in range(12):
    fig = wfdb.plot_items(
        # ecg_signal,
        ecg_signal_org[i],
        [rpeaks],
        fs=1,
        sig_name=["ECG"],
        sig_units=["mV"],
        time_units="seconds",
        return_fig=True,
        ann_style="o",
    )
    fig.show()
