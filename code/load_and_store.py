import numpy as np


def load_freq_volt():
    data_path_freq = './data/frequencies.npy'

    with open(data_path_freq, 'rb') as frequencies:
        freq_list = list(np.load(frequencies, allow_pickle=True))

    data_path_volt = './data/voltages.npy'
    with open(data_path_volt, 'rb') as voltages:
        volt_list = list(np.load(voltages, allow_pickle=True))

    return freq_list, volt_list
