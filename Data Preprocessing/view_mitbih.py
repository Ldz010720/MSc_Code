# 2.展示MIT-BIH原始ecg心电图，横坐标 point，纵坐标 V，需要有标题
# 3.展示MIT-BIH切割得到单片段心电图，横坐标 point，纵坐标 V，需要有标题
import wfdb
import matplotlib.pyplot as plt

pth = r'C:\Users\17673\Desktop\毕设\code\mit-bih-arrhythmia-database-1.0.0'

record = wfdb.rdrecord(f'{pth}/100', sampfrom=0, sampto=1000, physical=True, channels=[0, ])
signal_annotation = wfdb.rdann(f'{pth}/100', "atr", sampfrom=0, sampto=1000)
# 打印标注信息
ECG = record.p_signal

plt.figure()
plt.plot(ECG, label='ECG Signal')  # 添加 "ECG Signal" 标签
# 按坐标在散点图上绘点，统一星星颜色为橙色
for index in signal_annotation.sample:
    plt.scatter(index, ECG[index], marker='*', s=200, color='orange', label='Detected R-peaks' if index == signal_annotation.sample[0] else "")  # 只在第一个点上添加标签
plt.xlabel('Sampling point', fontsize=11)
plt.ylabel('Amplitude (mV)', fontsize=11)
plt.xticks(fontsize=9)  # 增大横坐标字体
plt.yticks(fontsize=9)  # 增大纵坐标字体

# 调整图例位置，缩小并放在图的右上角
plt.legend(loc='lower right', fontsize=9, bbox_to_anchor=(1, 1))

# plt.title('Segment Display of MIT-BIH ECG Dataset')
plt.show()


# xxx = signal_annotation.sample[3]
# y = ECG[xxx - 130:xxx + 130]
# plt.figure()
# plt.plot(y)
# plt.xlabel('Sampling point',fontsize=11)
# plt.ylabel('Amplitude (mV)',fontsize=11)
# plt.xticks(fontsize=9)  # 增大横坐标字体
# plt.yticks(fontsize=9)  # 增大纵坐标字体
# # plt.title('Single Normal beat ECG Signal from the MIT-BIH ECG Dataset')
# plt.show()

xxx = signal_annotation.sample[3]
y = ECG[xxx - 130:xxx + 130]
plt.figure()
plt.plot(y, label='ECG Signal')  # 添加标签
plt.xlabel('Sampling point', fontsize=11)
plt.ylabel('Amplitude (mV)', fontsize=11)
plt.xticks(fontsize=9)  # 增大横坐标字体
plt.yticks(fontsize=9)  # 增大纵坐标字体
plt.legend()  # 显示图例
# plt.title('Single Normal beat ECG Signal from the MIT-BIH ECG Dataset')  # 添加标题
plt.show()
