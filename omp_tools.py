import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


def generate_t_line(t_start=None, t_stop=None, t_num=None):
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


def cs_omp(dictionary, original_signal, sampling_matrix=None, mse_tolerance=1e-6, time_tolerance=600):
    # 使用omp算法，根据传入的原信号和字典，尝试使用字典元素的线性组合来复原原信号。
    # 可以选择传入一维采样阵，也可以不传入；不传入时会将原信号全部作为分析对象。
    # 可以选择传入mse（均方误差）容忍度，默认为1e-6。
    # 可以选择传入迭代次数容忍度，默认为600。
    # 最终返回三个变量：weight用于表示字典元素的线性组合权重，support用于表示参与组合的字典元素下标，time用于表示迭代总次数
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
    for time in tqdm(range(time_tolerance)):
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

        if np.linalg.norm(residual) < mse_tolerance:
            break

    return weight, support, time


def generate_fourier_element_in_dictionary(t, dictionary_scale, rank):
    frequencies = np.linspace(0, dictionary_scale - 1, dictionary_scale)
    return np.cos(2 * np.pi * frequencies[rank] * t)


def split_array(arr, y):
    indices = np.arange(len(arr))
    fragments = [arr[i:i + y] for i in indices[::y]]
    return fragments


def cs_huge_scale_omp(dictionary_scale, original_signal, t, generate_dictionary_element, sampling_matrix=None,
                      mse_tolerance=1e-6, time_tolerance=600, ram_usage_tolerance=None, ram_spare_tolerance=None,
                      picture_output=1):
    if sampling_matrix is not None:
        original_signal = original_signal * sampling_matrix

    # 初始化权重数组，用于表示字典中每个元素的权重
    weight = np.zeros(dictionary_scale)

    # 初始化残差信号，其初值为原始信号，在每一轮循环中会减去相关程度最高的字典信号
    residual = original_signal.copy()

    # 初始化支持集，其内容为已被选中的字典信号的索引
    support = []

    # 初始化迭代次数
    time = 0

    # 初始化字典取值
    dictionary = None
    dic_support = None
    mse = None

    # 迭代直至误差可容忍或迭代次数过多
    for time in tqdm(range(time_tolerance), disable=not bool(picture_output)):
        corr = np.zeros(dictionary_scale)

        # 计算每个序号对应的字典元素与当前残差的相关性
        for rank in range(dictionary_scale):
            if rank in support:
                pass
            else:
                dictionary_element = generate_dictionary_element(t, dictionary_scale, rank)
                if sampling_matrix is not None:
                    corr[rank] = (dictionary_element * sampling_matrix).T @ residual
                else:
                    corr[rank] = dictionary_element.T @ residual

        # 寻找最佳匹配元素
        i = np.argmax(np.abs(corr))

        # 将最佳匹配元素添加至Support
        support.append(i)

        # 生成临时字典
        dictionary = np.array([generate_dictionary_element(t, dictionary_scale, rank) for rank in support]).T
        dic_support = [i for i in range(len(support))]

        # 解决最小方差问题
        weight[dic_support] = np.linalg.lstsq(dictionary, original_signal, rcond=None)[0]

        # 更新残差
        residual = original_signal - dictionary @ weight[:len(dic_support)]

        # 记录迭代次数
        time += 1

        mse = np.linalg.norm(residual)
        if mse < mse_tolerance:
            break

    if picture_output == 1:
        draw_double_signal(t, original_signal, dictionary @ weight[:len(dic_support)])
        draw_single_signal(t, residual)
    recovered_signal = dictionary @ weight[:len(dic_support)]
    recover = [recovered_signal, dictionary, weight, time, mse]
    return recover


def count_mse(original_signal, recovered_signal):
    # 用于计算复原后信号与原始信号的最小均方误差
    return mean_squared_error(original_signal, recovered_signal)


def omp_terminal(weight, support, time, frequencies, mse=None, original_parameter=None):
    # 用于在当次omp迭代结束时在控制台输出结果
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

    if original_parameter is not None:
        print('原始信号数据：')
        original_parameter = sorted(original_parameter, key=lambda x: x[0], reverse=True)
        for cos in original_parameter[:5]:
            print(f'{cos[0]:.2f}'.strip(' ') + "*cos(" + f'{cos[1]:.2f}'.strip(' ') + "*2*pi*t)", end='')
            print('+', end='')
        print('\b')

    print(f'\nMSE：', mse)


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


def mse_terminal(MSE, mse_tolerance):
    # 用于输出多次运行后的MSE统计数据
    print('Max MSE:', np.max(MSE))
    print('Min MSE:', np.min(MSE))
    print('Average MSE:', np.mean(MSE))
    # 计算MSE数组中小于num_tolerance的元素所占的比例
    print('MSE <', mse_tolerance, ':', np.sum(np.log10(MSE) < np.log10(mse_tolerance)) / len(MSE))


def load_in_gray_image(path, picture_output=1):
    # 打开图像并转换为灰度图
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


def load_out_gray_image(path, img_array, resolution, picture_output=1):
    img_2d = img_array.reshape(resolution[0], resolution[1])
    if picture_output == 1:
        img = Image.fromarray(img_2d)
        if img.mode == "F":
            img = img.convert('L')
        img.show()
        img.save(path)
    return img_2d
