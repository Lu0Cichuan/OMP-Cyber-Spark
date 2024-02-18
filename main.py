import numpy as np

from tools import generate_fourier_dictionary_elements, cs_auto_romp

picture_load_in_path = 'pictures/raw'
picture_load_out_path = 'pictures/saved'
t = np.linspace(1, 10, 1000)
array = np.cos(t) + 2.2 * np.cos(10 * t)
recovered_array, mse = cs_auto_romp(array, 50, 30, generate_fourier_dictionary_elements,
                                    1e-6, 20, picture_output=1)

input("Press Enter to continue...")
