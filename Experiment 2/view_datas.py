import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from scipy import interpolate
import cv2

data = np.load('./my_data/01.npy')
print(data.shape)

kk = 2
data = data[kk]
print(data.shape)

i = 11
plt.figure()
plt.title(f"{i}")
plt.plot(data[i])
plt.show()

# for i in range(12):
#     plt.figure()
#     plt.title(f"{i}")
#     plt.plot(data[i])
#     plt.show()
