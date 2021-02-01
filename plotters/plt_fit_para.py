import matplotlib.pyplot as plt
import numpy as np


def plot_fit_params_f_z(fit_params_mat, freq_list, para_ind = 0):
    n_steps, n_f, _ = np.shape(fit_params_mat)

    z_vec = np.linspace(0, 1, n_steps)

    fig = plt.figure(figsize=(3, 2))

    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_f):
        ax.plot(z_vec, fit_params_mat[:, i, para_ind], label='f=' + str(freq_list[i]) + "Hz")
    ax.legend()




