# 4 展示12通道峰值检测的ecg心电图，横坐标 point，纵坐标 V，需要有标题
# 5.展示12通道切割得到单片段心电图，横坐标 point，纵坐标 V，需要有标题

import wfdb  # pip install wfdb
import glob
import numpy as np
import cv2
import neurokit2  # pip install neurokit2
import sleepecg  # pip install sleepecg
import wfdb.processing
import matplotlib.pyplot as plt

file_nm = '13'
root = rf'C:\Users\17673\Desktop\毕设\code\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords\{file_nm}'
save_name = f'data{file_nm}'

file_list = glob.glob(f"{root}/**/*.hea", recursive=True)
file_list = [x.rstrip('.hea') for x in file_list]
print(len(file_list))
file_list = file_list[:3]

pth = file_list[0]

record = wfdb.rdrecord(pth)
ventricular_signal = record.p_signal

ecg_signal = np.moveaxis(ventricular_signal, 0, -1)
ecg_signal = ecg_signal[:, :2000]
print(ecg_signal.shape)
# ecg_signal = cv2.resize(ventricular_signal, (3000, 12))
# print(ecg_signal.shape)
ecg_signal_org = ecg_signal
ecg_signal = ecg_signal[-2]  # 通道
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
plt.figure()
plt.plot(ecg_signal, label='ECG Signal')  # 添加 "ECG Signal" 标签
# 统一星星颜色为橙色，大小为200，只在第一个R峰上添加标签
for i, rpeak in enumerate(rpeaks):
    plt.scatter(rpeak, ecg_signal[rpeak], marker='*', s=200, color='orange', label='Detected R-peaks' if i == 0 else "")
plt.xlabel('Sampling point', fontsize=11)
plt.ylabel('Amplitude (mV)', fontsize=11)
plt.xticks(fontsize=9)  # 增大横坐标字体
plt.yticks(fontsize=9)  # 增大纵坐标字体

# 调整图例位置，缩小并放在图的右上角
plt.legend(loc='lower right', fontsize=9, bbox_to_anchor=(1, 1))

# plt.title('Segment of ECG Signal from One of the 12 Leads')
plt.show()


xxx = rpeaks[2]
y = ecg_signal[xxx - 130:xxx + 130]
plt.figure()
plt.plot(y, label='ECG Signal')  # 添加标签
plt.xlabel('Sampling point',fontsize=11)
plt.ylabel('Amplitude (mV)',fontsize=11)
plt.xticks(fontsize=9)  # 增大横坐标字体
plt.yticks(fontsize=9)  # 增大纵坐标字体
plt.legend()  # 显示图例
# plt.title('Single beat ECG Signal from the 12-Lead ECG Dataset')
plt.show()
