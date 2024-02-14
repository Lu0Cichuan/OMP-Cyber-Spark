import numpy as np
from tqdm import tqdm

from tools import draw_double_signal, split_array, generate_t_line, count_mse, draw_single_signal


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


def cs_huge_scale_omp(dictionary_scale, original_signal, t, generate_dictionary_element, sampling_matrix=None,
                      mse_tolerance=1e-6, time_tolerance=30, ram_usage_tolerance=None, ram_spare_tolerance=None,
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
        print(f'{mse = }')
    recovered_signal = dictionary @ weight[:len(dic_support)]
    recover = [recovered_signal, dictionary, weight, time, mse]
    return recover


def cs_pieces_omp(dictionary_scale, original_signal, pieces_length, t_range, generate_dictionary_element,
                  sampling_matrix=None,
                  mse_tolerance=1e-6, time_tolerance=600, ram_usage_tolerance=None, ram_spare_tolerance=None,
                  picture_output=1):
    truncated_arr = split_array(original_signal, pieces_length)
    solved_arr_list = []
    result = None
    t = None
    for i in tqdm(range(len(truncated_arr))):
        arr = truncated_arr[i]
        t = generate_t_line(0, t_range, arr.shape[0])
        solved_arr = cs_huge_scale_omp(dictionary_scale, arr, t, generate_dictionary_element, None,
                                       mse_tolerance, time_tolerance, picture_output=0)[0]
        # 将solved_arr添加到solved_arr_list中
        solved_arr_list.append(solved_arr)

        # 将solved_arr_list中的子数组拼接成一维数组
    result = np.concatenate(solved_arr_list)
    residual = original_signal - result
    mse = count_mse(original_signal, residual)
    t = generate_t_line(0, t_range, len(result))
    if picture_output == 1:
        draw_double_signal(t, original_signal, result)
    return result, mse


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


def mse_terminal(MSE, mse_tolerance):
    # 用于输出多次运行后的MSE统计数据
    print('Max MSE:', np.max(MSE))
    print('Min MSE:', np.min(MSE))
    print('Average MSE:', np.mean(MSE))
    # 计算MSE数组中小于num_tolerance的元素所占的比例
    print('MSE <', mse_tolerance, ':', np.sum(np.log10(MSE) < np.log10(mse_tolerance)) / len(MSE))
