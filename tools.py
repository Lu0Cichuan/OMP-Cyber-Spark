import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error


def generate_t_line(t_start=None, t_stop=None, t_num=None):
    # 生成一条时间轴
    ## 可以调整起始、终止与总采样点数
    if t_start is None and t_stop is None and t_num is None:
        t = np.linspace(0, 10, 5000, endpoint=False)
    elif (t_stop > t_start) and (t_num > 0):
        t = np.linspace(t_start, t_stop, t_num, endpoint=False)
    else:
        print('please enter valid time range and number of points')
        exit(-1)
    return t


def generate_random_cos_signal(t, frequencies, num):
    # 生成一些（个数为num）随机的余弦信号，并将它们叠加获得原始信号
    ## 暂不允许自定义原始信号
    signal = np.zeros(len(t))
    parameter = []
    for _ in range(num):
        strength = random.random()
        frequency = np.random.choice(frequencies)
        # np.append(frequencies, frequency)
        cos = strength * np.cos(2 * np.pi * frequency * t)
        signal = signal + cos
        parameter.append([strength, frequency])
    return signal, parameter


def generate_frequencies(frequencies_start=None, frequencies_stop=None, frequencies_interval=None):
    # 生成一个频率数组，用于生成随机余弦信号和字典
    ## 可以调整最低频率、最高频率和频率间隔
    if frequencies_start is None and frequencies_stop is None and frequencies_interval is None:
        frequencies = np.arange(0, 5, 0.001)
    elif (frequencies_start > frequencies_stop > 0) and (frequencies_interval > 0):
        frequencies = np.arange(frequencies_start, frequencies_stop, frequencies_interval)
    else:
        print('please enter valid frequencies range and interval')
        exit(-1)

    return frequencies


def generate_cos_dictionary(t, frequencies):
    # 生成字典
    dictionary = np.array([np.cos(2 * np.pi * f * t) for f in frequencies]).T
    return dictionary


def generate_sampling_matrix(t, gate):
    # 生成采样矩阵，用于对原始信号进行采样压缩
    ## 可以调整gate值，来控制采样阵中的0、1比例
    ## gate越大，0越多，1越少，有效采样越少，压缩率越高
    if gate is None:
        gate = 0.9
    sampling_matrix = np.zeros(t.shape)
    for i in range(len(t)):
        if random.random() < gate:
            sampling_matrix[i] = 0
        else:
            sampling_matrix[i] = 1

    return sampling_matrix


def cs_omp(dictionary, original_signal, sampling_matrix=None, num_tolerance=1e-2, time_tolerance=60):
    # 使用omp算法对被采样后的信号进行复原

    # 如果采样阵非空，则对字典和原信号均进行采样
    if sampling_matrix is not None:
        column_matrices = [dictionary[:, i] for i in range(dictionary.shape[1])]
        new_matrices = [column * sampling_matrix for column in column_matrices]
        dictionary = np.column_stack(new_matrices)
        original_signal = original_signal * sampling_matrix

    # 初始化权重数组，用于表示字典中每个元素的权重
    weight = np.zeros(dictionary.shape[1])

    # 初始化残差信号，其初值为原始信号，在每一轮循环中会减去相关程度最高的字典信号
    residual = original_signal.copy()

    # 初始化支持集，其内容为已被选中的字典信号的索引
    support = []

    # 初始化迭代次数
    time = 0

    # 迭代直至误差可容忍或迭代次数过多
    while np.linalg.norm(residual) > num_tolerance and time < time_tolerance:
        # 计算相关性
        corr = dictionary.T @ residual

        # 寻找最佳匹配元素
        i = np.argmax(np.abs(corr))

        # 将最佳匹配元素添加至Support
        support.append(i)

        # 解决最小方差问题
        weight[support] = np.linalg.lstsq(dictionary[:, support], original_signal, rcond=None)[0]

        # 更新残差
        residual = original_signal - dictionary @ weight

        # 记录迭代次数
        time += 1

    return weight, support, time


def count_mse(original_signal, recovered_signal):
    # 计算复原后信号与原始信号的最小均方误差
    return mean_squared_error(original_signal, recovered_signal)


def omp_terminal(weight, support, time, frequencies, original_parameter, mse):
    # 在控制台输出当次OMP迭代运行结果
    print(f'\nOMP迭代完成。总次数:', time)

    print('当前迭代结果：')
    for i in support[:5]:
        print(f"{float(str(weight[i])[:4].strip('[').strip(']')):.2f}" +
              "*cos(" +
              f'{frequencies[i]:.2f}' +
              "*2*pi*t)",
              end='')
        print('+', end='')
    print('\b')

    print('原始信号数据：')
    original_parameter = sorted(original_parameter, key=lambda x: x[0], reverse=True)
    for cos in original_parameter[:5]:
        print(f'{cos[0]:.2f}'.strip(' ') + "*cos(" + f'{cos[1]:.2f}'.strip(' ') + "*2*pi*t)", end='')
        print('+', end='')
    print('\b')

    print(f'\nMSE：', mse)


def draw_single_signal(x, y, title, xlabel=None, ylabel=None):
    # 画一条线
    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def draw_double_signal(x, y1, y2, title, xlabel=None, ylabel=None):
    # 画两条相互对比的线
    plt.figure(figsize=(10, 4))
    plt.plot(x, y1, label='Original')
    plt.plot(x, y2, label='Reconstructed')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()
