import matplotlib.pyplot as plt
import numpy as np


def omp(dictionary, original_signal, frequencies, num_tolerance=1e-6, time_tolerance=60,
        picture_output=1, terminal_output=1, log_output=1, ):
    """
    Orthogonal Matching Pursuit (OMP)
    dictionary: 字典
    original_signal: 原始信号
    tol: tolerance、容忍度
    sampling:采样压缩矩阵
    """
    # 初始化

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

        # 输出日志
        if terminal_output == 1:
            omp_terminal(weight, support, time, frequencies)
    if terminal_output == 1:
        print('OMP迭代完成。总次数:', time)
        print('当前迭代结果：')
        for i in support:
            print(str(weight[i])[:4].strip('[').strip(']') +
                  "*cos(" +
                  str(frequencies[i]).strip('[').strip(']') +
                  "0*2*pi*t)",
                  end='')
            print('+', end='')
        print('\b')
    return weight


def cs_omp(dictionary, original_signal, frequencies, num_tolerance=1e-6, time_tolerance=60,
           picture_output=1, terminal_output=1, log_output=1, sampling=None, t=None):
    """
    Orthogonal Matching Pursuit (OMP)
    dictionary: 字典
    original_signal: 原始信号
    tol: tolerance、容忍度
    sampling:采样压缩矩阵
    """
    # 初始化

    # 如果采样阵非空，则对字典和原信号均进行采样
    if sampling is not None:
        column_matrices = [dictionary[:, i] for i in range(dictionary.shape[1])]
        new_matrices = [column * sampling for column in column_matrices]
        dictionary = np.column_stack(new_matrices)
        original_signal = original_signal * sampling

    # 画出采样后的信号
    if t is not None and picture_output == 1:
        draw_single_signal(t, original_signal, 'Sampled signal', 'Time', 'Amplitude')

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

        # 输出终端信息
        if terminal_output == 1:
            omp_terminal(weight, support, time, frequencies)

    if terminal_output == 1:
        print('OMP迭代完成。总次数:', time)
        print('当前迭代结果：')
        for i in support:
            print(str(weight[i])[:4].strip('[').strip(']') +
                  "*cos(" +
                  str(frequencies[i]).strip('[').strip(']') +
                  "0*2*pi*t)",
                  end='')
            print('+', end='')
        print('\b')
    return weight


def omp_terminal(weight, support, time, frequencies):
    print("当前迭代次数：" + str(time).strip('[').strip(']') + '。结果如下：')
    print("新增支撑集：cos(" + str(frequencies[[support[-1]]]).strip('[').strip(']') + "0*2*pi*t)")
    print("对应系数：" + str(weight[support[-1]]).strip('[').strip(']'))
    print('')


def draw_single_signal(x, y, title, xlabel=None, ylabel=None):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def draw_double_signal(x, y1, y2, title, xlabel=None, ylabel=None):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y1, label='Original')
    plt.plot(x, y2, label='Reconstructed')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()
