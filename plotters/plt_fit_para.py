import matplotlib.pyplot as plt
import numpy as np


def plot_depth_f_z(fit_params_mat, freq_list):
    n_steps, n_f, _ = np.shape(fit_params_mat)

    z_vec = np.linspace(0, 1, n_steps)

    fig = plt.figure(figsize=(3, 2))

    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_f):
        ax.plot(z_vec, fit_params_mat[:, i, 1], label='f=' + str(freq_list[i]) + "Hz")
    ax.legend()


def plot_current_f_z(fit_params_mat, freq_list):
    n_steps, n_f, _ = np.shape(fit_params_mat)

    z_vec = np.linspace(0, 1, n_steps)
    fig = plt.figure(figsize=(3, 2))

    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_f):
        ax.plot(z_vec, fit_params_mat[:, i, 0], label='f=' + str(freq_list[i]) + "Hz")
    ax.legend()
