import json
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
import psutil
from PIL import Image
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def generate_t_line(t_start, t_stop, t_num):
    # 这一函数用于生成一条时间轴，以一维np数组形式表示。需要定义起点、终点与总采样数。
    if t_start is None and t_stop is None and t_num is None:
        t = np.linspace(0, 10, 5000, endpoint=False)
    elif (t_stop > t_start) and (t_num > 0):
        t = np.linspace(t_start, t_stop, t_num, endpoint=False)
    else:
        print('please enter valid time range and number of points')
        exit(-1)
    return t


def generate_random_signal(signal_type, t, frequency=2, log_base=10, power_base=3, strength=None):
    # 这一函数用于生成各种形式的随机信号。当前支持的类型有正余弦、指数、对数与幂函数；必须传入的参数是信号类型与时间轴，以及信号类型对应的特定参数。
    # 正余弦信号需要额外传入圆频率，指数信号不需要额外参数，对数信号需要额外传入底数，幂函数信号需要额外传入幂值。
    # 对于信号前的系数，若未指定，默认为在0~1中随机取值。
    if strength is None:
        strength = random.random()
    if signal_type == 'cos' or signal_type == 'sin':
        try:
            return strength * np.cos(2 * np.pi * frequency * t)
        except:
            print('please enter valid parameter to generate signal.')
            exit(-1)
    elif signal_type == 'e':
        try:
            return strength * np.exp(t)
        except:
            print('please enter valid parameter to generate signal.')
            exit(-1)
    elif signal_type == 'log':
        try:
            return strength * np.log10(t) / np.log10(log_base)
        except:
            print('please enter valid parameter to generate signal.')
            exit(-1)
    elif signal_type == 'power':
        try:
            return strength * np.power(t, power_base)
        except:
            print('please enter valid parameter to generate signal.')
            exit(-1)
    else:
        print('please enter valid signal type to generate signal.')


def generate_fourier_dictionary(t, frequencies=None):
    # 这一函数用于生成傅里叶字典。必须传入时间轴，可以选择传入包含所有所需频率的数组，也可以不传入；不传入时根据时间轴自动生成字典。
    if frequencies is None:
        frequencies = np.linspace(0, t[-1], t.shape[0] // 10)
    dictionary = np.array([np.cos(2 * np.pi * f * t) for f in frequencies]).T
    return dictionary


def generate_sampling_line(t, gate):
    # 这一函数用于生成一维采样阵，模拟单光子成像系统中单个光子发射并返回、对单个像素点进行探测的物理过程。
    # 需要传入时间轴和采样门限。由于对每一个点而言，是否采样均是由随机数是否大于门限值而决定的；因此采样门限越高，采样点越少。
    if gate is None:
        gate = 0.9
    sampling_matrix = np.zeros(t.shape)
    for i in range(len(t)):
        if random.random() < gate:
            sampling_matrix[i] = 0
        else:
            sampling_matrix[i] = 1

    return sampling_matrix


def generate_fourier_dictionary_elements(t, dictionary_scale, rank):
    frequencies = np.linspace(0, (dictionary_scale - 1) / 10, dictionary_scale)
    return np.cos(2 * np.pi * frequencies[rank] * t)


def split_array(arr, y):
    indices = np.arange(len(arr))
    fragments = [arr[i:i + y] for i in indices[::y]]
    return fragments


def count_mse(original_signal, recovered_signal):
    # 用于计算复原后信号与原始信号的最小均方误差
    return mean_squared_error(original_signal, recovered_signal)


def draw_single_signal(x, y, title=None, xlabel=None, ylabel=None):
    # 画一条线
    plt.figure(figsize=(10, 4))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def draw_double_signal(x, y1, y2, title=None, xlabel=None, ylabel=None):
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


def draw_triple_signal(x, y1, y2, y3, title=None, xlabel=None, ylabel=None):
    # 画两条相互对比的线
    plt.figure(figsize=(10, 4))
    plt.plot(x, y1, label='Original')
    plt.plot(x, y2, label='Recovered')
    plt.plot(x, y3, label='Residual')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.show()


def draw_mse_line(t, MSE):
    # 绘制连续多次运行时mse随t的变化曲线
    ## 纵坐标被设置为对数坐标
    ## 增补了代表yy平均值的虚线
    plt.figure()
    plt.plot(t, MSE, label='MSE')
    plt.yscale('log')
    plt.xlabel('t')
    plt.ylabel('MSE')
    plt.title('MSE vs t')
    plt.legend()

    avg_yy = np.mean(MSE)
    plt.axhline(y=avg_yy, color='r', linestyle='--', label='Average MSE')
    plt.legend()

    plt.show()


def load_in_gray_image(path, rank=None, picture_output=1):
    # 打开图像并转换为灰度图
    if rank is None:
        img = Image.open(path).convert('L')
    else:
        path = path + '/' + str(rank) + '.jpg'
        img = Image.open(path).convert('L')

    # 判断是否为灰度图
    if img.mode == 'L':
        is_grayscale = 0
    else:
        is_grayscale = 1

    # 获取图像分辨率
    resolution = [img.height, img.width]

    # 转换为一维数组
    img_array = np.array(img).flatten()

    # 显示图像
    if picture_output == 1:
        img.show()

    img = [img_array, resolution, is_grayscale]
    return img


def load_out_gray_image(path, img_array, resolution, rank=None, picture_output=1):
    img_2d = img_array.reshape(resolution[0], resolution[1])
    img = Image.fromarray(img_2d)
    if img.mode == "F":
        img = img.convert('L')
    if picture_output == 1:
        img.show()

    if rank is None:
        pass
    else:
        path = path + '/' + str(rank) + '.jpg'

    i = 0
    while True:
        if os.path.exists(path + '_' + str(i)):
            i = i + 1
            pass
        else:
            img.save(path + '_' + str(i))
            break
    return img_2d


def divide_and_round_up(x, y):
    quotient = x // y
    remainder = x % y

    if remainder != 0:
        quotient += 1

    return quotient


def cs_auto_omp(original_signal, pieces_length, dictionary_scale, generate_dictionary_element,
                mse_tolerance=1e-6, time_tolerance=20, sampling_matrix=None,
                t_range=1,
                picture_output=1):
    if isinstance(original_signal, str):
        if original_signal[-4:0] != 'json':
            print('要求载入的文件格式不受支持。请载入json格式文件。')
            exit(-1)
        if os.path.exists(original_signal):
            with open(original_signal, 'r') as file:
                json_data = json.load(file)
                signal = np.array(json_data).flatten()
            print('已成功从' + original_signal + '载入数据。')
            print('数据长度为' + str(len(signal)) + '。')
        else:
            print('未找到指定文件，请重新确认文件路径。')
            exit(-2)
    elif isinstance(original_signal, np.ndarray):
        signal = original_signal.flatten()
        print('已成功从数组载入数据。')
        print('数据长度为' + str(len(signal)) + '。')
    else:
        print('进行了无效的导入数据操作。请确认您的导入内容为json路径或numpy数组。')
        exit(-3)

    if sampling_matrix is None:
        sampling_operation = 0
    elif isinstance(sampling_matrix, np.ndarray):
        sampling_matrix = sampling_matrix.flatten()
        print('采样矩阵已导入。规模为' + str(len(sampling_matrix)) + '。')
        sampling_operation = 1
    else:
        print('请导入np.ndarray类型的采样矩阵。')
        exit(-4)

    if pieces_length <= 0 or pieces_length is None or pieces_length >= len(signal):
        if sampling_operation == 1 and len(sampling_matrix) != len(signal):
            print('采样矩阵规模与切片长度不符。')
            print('采样矩阵规模为' + str(len(sampling_matrix)) + '，切片长度为' + str(pieces_length) + '。')
            exit(-5)
        else:
            pieces_disable = 1
            print('正在执行不切片复原算法。')
    else:
        if sampling_operation == 1 and len(sampling_matrix) != pieces_length:
            print('采样矩阵规模与信号长度不符。')
            print('采样矩阵规模为' + str(len(sampling_matrix)) + '，信号长度为' + str(len(signal)) + '。')
            exit(-6)
        else:
            pieces_disable = 0
            print('正在执行分片复原算法。单个切片长度为' + str(pieces_length) + ",预计切片总数为" +
                  str(divide_and_round_up(len(signal), pieces_length)) + '。')

    if pieces_disable == 0:
        pieced_arrays = split_array(original_signal, pieces_length)
    else:
        pieced_arrays = [signal]
    solved_list = []
    print('已开始复原信号。')
    for i in tqdm(range(len(pieced_arrays)), disable=bool(pieces_disable)):
        arr = pieced_arrays[i]
        t = generate_t_line(random.random() - 1, t_range, arr.shape[0])
        weight = np.zeros(dictionary_scale)

        # 初始化残差信号，其初值为原始信号，在每一轮循环中会减去相关程度最高的字典信号
        residual = arr.copy()

        # 初始化支持集，其内容为已被选中的字典信号的索引
        support = []

        # 迭代直至误差可容忍或迭代次数过多
        for time in range(time_tolerance):
            corr = np.zeros(dictionary_scale)
            if sampling_operation == 1:
                sampling_matrix = sampling_matrix[:len(arr)]
            dictionary_store = {}
            # 计算每个序号对应的字典元素与当前残差的相关性
            for rank in range(dictionary_scale):
                system_memory_info = psutil.virtual_memory()
                Memory_Available = system_memory_info.available / 1024 / 1024
                if rank in dictionary_store.keys():
                    dictionary_element = dictionary_store[rank]
                elif Memory_Available > 2048:
                    dictionary_element = generate_dictionary_element(t, dictionary_scale, rank)
                    dictionary_store[rank] = dictionary_element
                else:
                    dictionary_element = generate_dictionary_element(t, dictionary_scale, rank)

                if sampling_operation == 1:
                    corr[rank] = (dictionary_element * sampling_matrix).T @ residual
                else:
                    corr[rank] = dictionary_element.T @ residual

            # 寻找最佳匹配元素
            corr_max = np.max(np.abs(corr))
            append_support = [index for index, value in enumerate(corr) if np.abs(value) > 0.8 * corr_max]
            support = list(set(append_support + support))

            # 生成临时字典
            temporary_dictionary = np.array(
                [generate_dictionary_element(t, dictionary_scale, rank) for rank in support]).T
            if sampling_operation == 1:
                temporary_sampling_matrix = np.broadcast_to(sampling_matrix.T, (len(support), len(sampling_matrix)))
                temporary_dictionary = temporary_sampling_matrix.T * temporary_dictionary

            # 解决最小方差问题
            weight[support] = np.linalg.lstsq(temporary_dictionary, arr, rcond=None)[0]
            temporary_weight = np.zeros(len(support))
            for j in range(len(support)):
                temporary_weight[j] = weight[support[j]]

            # 更新残差
            residual = arr - temporary_dictionary @ temporary_weight[:len(support)]

            # 记录迭代次数
            time += 1

            mse = np.linalg.norm(residual)
            if mse < mse_tolerance:
                break

        recovered_signal = np.zeros(len(arr))
        for rank in support:
            recovered_signal = recovered_signal + dictionary_store[rank] * weight[rank]

        # 将solved_arr添加到solved_arr_list中
        solved_list.append(recovered_signal)

    # 将solved_arr_list中的子数组拼接成一维数组
    result = np.concatenate(solved_list)
    residual = original_signal - result
    mse = count_mse(original_signal, result)
    t = generate_t_line(0, t_range, len(result))
    if picture_output == 1:
        draw_triple_signal(t, original_signal, result, np.abs(residual))

    return result, mse
    # 初始化权重数组，用于表示字典中每个元素的权重
