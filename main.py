import numpy as np
from tqdm import tqdm

import omp_tools as omp

img = omp.load_in_gray_image('pictures/raw/2.JPG', 1)
img_array = img[0]

length = 20
truncated_arr = omp.split_array(img_array, length)
solved_arr_list = []
result = None
for i in tqdm(range(len(truncated_arr))):
    # for arr in truncated_arr:
    arr = truncated_arr[i]
    # 对每个子项进行操作，得到solved_arr
    t = omp.generate_t_line(0, 4, arr.shape[0])
    solved_arr = omp.cs_huge_scale_omp(200, arr, t, omp.generate_fourier_element_in_dictionary,
                                       time_tolerance=10, picture_output=0)[0]
    # 将solved_arr添加到solved_arr_list中
    solved_arr_list.append(solved_arr)

    # 将solved_arr_list中的子数组拼接成一维数组
    result = np.concatenate(solved_arr_list)
omp.load_out_gray_image('pictures/saved/2.JPG', result, resolution=img[1], picture_output=1)
input("Press Enter to continue...")
