import os

from tools import load_in_gray_image, generate_fourier_element_in_dictionary, load_out_gray_image, cs_pieces_omp

picture_rank = 2
img = load_in_gray_image('pictures/raw/' + str(picture_rank) + '.JPG', 1)
img_array = img[0]

recovered_img_array, mse = cs_pieces_omp(40, img_array, 30, 0.5, generate_fourier_element_in_dictionary,
                                         None, mse_tolerance=1e-6, time_tolerance=20, picture_output=1)
i = 0
while True:
    if os.path.exists('pictures/saved/' + str(picture_rank) + '_' + str(i) + '.JPG'):
        i = i + 1
        pass
    else:
        load_out_gray_image('pictures/saved/' + str(picture_rank) + '_' + str(i) + '.JPG', recovered_img_array,
                            resolution=img[1], picture_output=1)
        break
input("Press Enter to continue...")
