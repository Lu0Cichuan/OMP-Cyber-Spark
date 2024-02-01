import random

import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from mp import cs_omp, draw_single_signal, draw_double_signal


def settings(t=None, frequencies=None,
             gate=None,
             picture_output=0, terminal_output=0, log_output=0):
    if t is None:
        t = np.linspace(0, 10, 5000, endpoint=False)
    # 设置时间轴
    ## 可以调整起始、终止与总取点数

    if frequencies is None:
        frequencies = np.arange(0, 5, 0.001)
    # 设置字典频率范围，生成字典
    ## 可以调整字典的频率范围
    ## 务必保证字典中的元素是可以完全表示信号的

    if gate is None:
        gate = 0.99
    # 设置采样阵gate值
    ## 可以调整gate值，用于控制sampling中的0、1比例
    ## gate越大，0越多，1越少，有效采样越少，压缩率越高

    if picture_output is None:
        picture_output = 1
    if terminal_output is None:
        terminal_output = 1
    if log_output is None:
        log_output = 1
    # 设置运行过程中是否输出图片、终端信息、日志信息
    ## 正常运行请填1开启，机器学习或处理大量数据时请填0关闭

    return t, frequencies, gate, picture_output, terminal_output, log_output


def initialize(t, frequencies, gate):
    cos1 = 1 * np.cos(2 * np.pi * 2.79 * t)
    cos2 = 1.9 * np.cos(2 * np.pi * 3.22 * t)
    cos3 = 2.784 * np.cos(2 * np.pi * 1.45 * t)
    # 生成三个不同频率的余弦信号
    ## 可以调整每个信号的振幅、频率
    ## 也可以自己添加新的信号

    signal = cos1 + cos2 + cos3
    # 将这三个信号叠加起来

    D = np.array([np.cos(2 * np.pi * f * t) for f in frequencies]).T
    # 生成字典

    sampling = np.zeros(t.shape)
    for i in range(len(t)):
        if random.random() < gate:
            sampling[i] = 0
        else:
            sampling[i] = 1
    # 创建采样矩阵

    return signal, D, sampling


def main():
    t, frequencies, gate, picture_output, terminal_output, log_output = settings()
    # 根据设置函数，初始化运行基本参数

    signal, D, sampling = initialize(t, frequencies, gate)
    # 初始化原信号、字典与采样阵

    if picture_output == 1:
        draw_single_signal(t, signal, 'Original Signal', 'Time', 'Amplitude')
    # 绘制生成的信号

    weight_rec = cs_omp(D, signal, frequencies, 1e-3, 20,
                            picture_output, terminal_output, log_output, sampling, t)

    signal_rec = D @ weight_rec
    # 计算重建后的信号

    mse = mean_squared_error(signal, signal_rec)
    if terminal_output == 1:
        print('均方误差:', mse)
    # 计算均方误差

    if picture_output == 1:
        draw_double_signal(t, signal, signal_rec, 'Signal Recovery with CS_OMP', 'Time', 'Amplitude')
    # 画出重建信号与原始信号进行比对

    return mse


def test(orientation=10):
    MSE = []
    for _ in tqdm(range(orientation), desc='OMP Running'):
        MSE.append(main())
    print('MSE均值:', np.mean(MSE))
    print('MSE大于1的次数:', np.sum(np.array(MSE) > 1))
    print('信号恢复率：', 100 * (1 - np.sum(np.array(MSE) > 1) / orientation), '%')


test(100)
