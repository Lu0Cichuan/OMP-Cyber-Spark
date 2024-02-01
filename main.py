from tools import generate_random_cos_signal, generate_cos_dictionary, generate_sampling_matrix, \
    generate_t_line, generate_frequencies, cs_omp, count_mse, omp_terminal, draw_single_signal, draw_double_signal


def main():
    num = 4
    gate = 0.94
    picture_output = 1
    terminal_output = 1
    log_output = 1
    # 设置运行过程中是否输出图片、终端信息、日志信息
    ## 正常运行请填1开启，机器学习或处理大量数据时请填0关闭

    t = generate_t_line(t_start=None, t_stop=None, t_num=None)

    frequencies = generate_frequencies(frequencies_start=None, frequencies_stop=None, frequencies_inteval=None)

    original_signal, original_parameter = generate_random_cos_signal(t, frequencies, num)

    if picture_output == 1:
        draw_single_signal(t, original_signal, "Original Signal")

    dictionary = generate_cos_dictionary(t, frequencies)

    sampling_matrix = generate_sampling_matrix(t, gate)

    recovered_weight, support, time = cs_omp(dictionary, original_signal, sampling_matrix, 1, 20)

    recovered_signal = dictionary @ recovered_weight

    mse = count_mse(recovered_signal, original_signal)

    if terminal_output == 1:
        omp_terminal(recovered_weight, support, time, frequencies, original_parameter, mse)

    if picture_output == 1:
        draw_single_signal(t, original_signal * sampling_matrix, 'Sampled Signal')
        draw_double_signal(t, original_signal, recovered_signal, "Original Signal and Recovered Signal")


main()
