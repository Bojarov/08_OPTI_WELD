import numpy as np


def load_freq_volt():
    data_path_freq = './data/frequencies.npy'

    with open(data_path_freq, 'rb') as frequencies:
        freq_list = list(np.load(frequencies, allow_pickle=True))

    data_path_volt = './data/voltages.npy'
    with open(data_path_volt, 'rb') as voltages:
        volt_list = list(np.load(voltages, allow_pickle=True))

    return freq_list, volt_list

def load_array(folder_path, array_name):
    with open(folder_path+array_name+'.npy', 'rb') as np_mat:
        np_arr = np.load(np_mat, allow_pickle=True)

    return np_arr

def save_array(folder_path, array_name, array):
    with open(folder_path+array_name+'.npy', 'wb') as np_mat:
        np.save(np_mat, array, allow_pickle=True)
