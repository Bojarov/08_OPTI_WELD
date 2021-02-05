import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import code.data_sym as ds


def plot_bare_signal_symmetry_f_z(volt_list, freq_list, det_ind, n_win):
    """
    plots the measured b field y-component difference between detectors
    located at opposite sides of the detector array, for every measure point
    det_ind: picks the detector pair: 0 most outer pair
                                     1 is next inner pair ...
    n_win: determines the window for the plot of the rolling average along pipe axis
    """

    n_steps, n_det, n_f, _ = np.shape(volt_list)

    z_vec = np.linspace(0, 1, n_steps)
    fig = plt.figure(figsize=(3, 2))

    ax1 = fig.add_subplot(2, 1, 1)

    y_sym_mat = ds.by_sym_mat(volt_list, det_ind)
    for i in range(n_f):
        ax1.plot(z_vec, y_sym_mat[:, i], linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)

    ax1.legend()
    ax1.set_xlim(0.0, 1.0)

    ax2 = fig.add_subplot(2, 1, 2)
    for i in range(n_f):
        f_y_sym = pd.Series(y_sym_mat[:, i])
        f_y_sym_roll = np.array(f_y_sym.rolling(n_win, center=True).sum()) / n_win
        ax2.plot(z_vec, f_y_sym_roll, linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)
    ax2.legend()
    ax2.set_xlim(0.0, 1.0)



def plot_bare_signal_symmetry_norm_f_z(volt_list, freq_list, fit_params_mat, n_win, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    z_vec = np.linspace(0, 1, n_steps)
    fig = plt.figure(figsize=(3, 2))

    ax1 = fig.add_subplot(2, 1, 1)

    y_sym_mat = (np.array(volt_list)[:, 0, :, 1] - np.array(volt_list)[:, -1, :, 1]) / (
            fit_params_mat[:, :, 0] * factor)

    for i in range(n_f):
        ax1.plot(z_vec, y_sym_mat[:, i], linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)

    ax1.legend()
    ax1.set_xlim(0.0, 1.0)

    ax2 = fig.add_subplot(2, 1, 2)
    for i in range(n_f):
        f_y_sym = pd.Series(y_sym_mat[:, i])
        f_y_sym_roll = np.array(f_y_sym.rolling(n_win, center=True).sum()) / n_win
        ax2.plot(z_vec, f_y_sym_roll, linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)
    ax2.legend()
    ax2.set_xlim(0.0, 1.0)
