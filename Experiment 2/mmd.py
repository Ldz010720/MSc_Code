import numpy as np


def gaussian_kernel(x, y, sigma=1.0):
    """
    计算高斯核函数的值。

    参数:
    - x, y: 输入向量。
    - sigma: 高斯核的带宽参数。

    返回:
    - 高斯核函数在x和y上的值。
    """
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * sigma ** 2))


def mmd_rbf(X, Y, sigma=1.0):
    """
    使用高斯核计算两个样本集之间的最大均值差异（MMD）。

    参数:
    - X, Y: 两个样本集的numpy数组。
    - sigma: 高斯核的带宽参数。

    返回:
    - X和Y之间的MMD值。
    """
    XX = np.mean([gaussian_kernel(x, x_prime, sigma) for x in X for x_prime in X])
    YY = np.mean([gaussian_kernel(y, y_prime, sigma) for y in Y for y_prime in Y])
    XY = np.mean([gaussian_kernel(x, y, sigma) for x in X for y in Y])

    return XX + YY - 2 * XY

if __name__=="__main__":
    X = np.random.normal(0, 1, (100, 2))  # 从分布P生成的样本
    Y = np.random.normal(0.5, 1.5, (100, 2))  # 从分布Q生成的样本

    # 计算MMD
    mmd_value = mmd_rbf(X, Y)
    print("MMD:", mmd_value)
