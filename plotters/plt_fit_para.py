import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import pandas as pd


def plot_fit_params_f_z(fit_params_mat, freq_list, para_ind=0):
    n_steps, n_f, _ = np.shape(fit_params_mat)

    z_vec = np.linspace(0, 1, n_steps)

    fig = plt.figure(figsize=(3, 2))

    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_f):
        ax.plot(z_vec, fit_params_mat[:, i, para_ind], label='f=' + str(freq_list[i]) + "Hz")
    ax.legend()

    ax.set_xlabel(r'$z$', fontsize=20)
    ax.set_ylabel(r'$x_0 $', fontsize=20)
    plt.grid(True)
    #ax.set_xlim(0, 100)


def plot_dist_fit_params(fit_params_mat, freq_list, para_ind=0):
    n_steps, n_f, _ = np.shape(fit_params_mat)

    fig = plt.figure(figsize=(6, 2))
    ax1 = fig.add_subplot(1, 3, 1)
    sample = fit_params_mat[:, 0, para_ind]

    n, bins, patches = plt.hist(sample, 25, weights=np.ones_like(sample) / len(sample),
                                facecolor='blue', alpha=0.75)
    print(abs(bins[0] - bins[1]))

    (mu, sigma) = stats.norm.fit(sample)
    print(sigma / mu)
    y = norm.pdf(np.linspace(-0.05, 0.05), mu, sigma) * abs(bins[0] - bins[1])
    plt.grid(True)
    title = r'Fit results: $\mu$ = %.5f,  $\sigma$ = %.5f' % (mu, sigma)
    plt.title(title)
    ax1.plot(np.linspace(-0.05, 0.05), y, 'r', linewidth=2)
    ax1.set_ylabel("relative frequency")
    ax1.set_xlabel(r'$x_s$[m]')
    ax1.set_xlim(-0.05, 0.05)

    ax2 = fig.add_subplot(1, 3, 2)
    for i in range(5):
        sample = fit_params_mat[i * 20:(i + 1) * 19, 0, para_ind]
        (mu, sigma) = stats.norm.fit(sample)
        y = norm.pdf(np.linspace(-0.05, 0.05), mu, sigma) * abs(bins[0] - bins[1])
        ax2.plot(np.linspace(-0.05, 0.05), y, linewidth=2, label=str(i))
    ax2.legend()

    ax3 = fig.add_subplot(1, 3, 3)
    n_win = 10
    for i in range(n_f):
        fit_f_z = pd.Series(fit_params_mat[:, i, para_ind])
        fit_f_z_roll = np.array(fit_f_z.rolling(n_win, center=True).sum()) / n_win
        ax3.plot(-fit_f_z_roll, linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)
    ax3.set_xlim(0, 100)
    ax3.legend()


def plot_fit_sym_comp(volt_list, fit_params_mat_1, fit_params_mat_2, freq_list, para_ind=2, factor=89000):
    # asymmetry in bare field data normalized by current
    n_f= len(freq_list)
    y_sym_mat = (np.array(volt_list)[:, 0, :, 1] - np.array(volt_list)[:, -1, :, 1]) / (
            fit_params_mat_1[:, :, 0] * factor)




    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 1, 1)

    n_win = 1
    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))
    for i in range(n_f):
        c = next(color)
        fit_f_z = pd.Series(fit_params_mat_2[:, i, para_ind])
        fit_f_z_roll = np.array(fit_f_z.rolling(n_win, center=True).sum()) / n_win
        fit_f_z_roll = fit_f_z_roll / np.nanmax(abs(fit_f_z_roll))

        sym_f_z = pd.Series(y_sym_mat[:, i])
        sym_f_z_roll = np.array(sym_f_z.rolling(n_win, center=True).sum()) / n_win
        sym_f_z_roll = sym_f_z_roll / np.nanmax(abs(sym_f_z_roll))

        #ax.plot((abs(fit_f_z_roll)-abs(sym_f_z_roll))/abs(sym_f_z_roll), linestyle='-', color=c, label='f=' + str(freq_list[i]) + "Hz")
        ax.plot((-fit_f_z_roll/abs(sym_f_z_roll)), linestyle='-', color=c, label='f=' + str(freq_list[i]) + "Hz")
        ax.plot(sym_f_z_roll/sym_f_z_roll, linestyle='--', color='black')
        ax.plot(-sym_f_z_roll/sym_f_z_roll, linestyle='--', color='black')
        # ax.plot(sym_f_z_roll, linestyle='--', color=c) # , label='f=' + str(freq_list[i]) + "Hz")
    ax.set_xlabel(r'$z$', fontsize=20)
    ax.set_ylabel(r'$\Delta  \mathcal{B}_x/\left|\mathcal{X}_0\right| $', fontsize=20)
    plt.grid(True)
    ax.set_xlim(0, 100)
    ax.legend()
