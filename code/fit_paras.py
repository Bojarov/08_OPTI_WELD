from inspect import signature
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def fit_params(my_func, x_data, volt_list, factor=89000):
    sig = signature(my_func)
    n_steps, _, n_f, _ = np.shape(volt_list)
    n_fit_params = len(list(sig.parameters)) - 1

    fit_params_mat = np.zeros((n_steps, n_f, n_fit_params))

    for i in range(n_steps):
        for j in range(n_f):
            y_data = np.array(volt_list)[i, :, j, 1] / factor

            p_opt, _ = curve_fit(my_func, x_data, y_data)


            fit_params_mat[i, j, :] = p_opt
    return fit_params_mat


def fit_params_FH_data(my_func, factor=89000):
    bvec = np.load('./bvec2_l20.npy')
    bvec_i = np.load('./bvec_i2_l20.npy')
    sig = signature(my_func)
    n_fit_params = len(list(sig.parameters)) - 1
    n_steps = 1
    n_f, n_det = np.shape(bvec)

    freq_list = [3, 6, 12]

    fit_params_mat = np.zeros((n_steps, n_f, n_fit_params))
    x_data = np.array([-1.0, -0.75, -0.5, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.5, 0.75, 1.0])
    x_fit = np.linspace(-0.5,0.5,100)
    i_ideal_vec =np.array([2.6420795743181618, 1.3212431653912475, 0.6608175584692688])
    i_damage_vec =np.array([2.641187476588734, 1.3208711101054076, 0.6606485501499122])
    for i in range(n_steps):
        for j in range(n_f):
            y_data = bvec[j, :] * i_damage_vec[j] *2*10**(-7) / (1000)  # np.array(volt_list)[i, :, j, 1] / factor
            #print(y_data)

            #print(x_data)
            #print(y_data)

            p_opt, _ = curve_fit(my_func, x_data, y_data)

            fit_params_mat[i, j, :] = p_opt
            print(p_opt)


    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(n_f):


        p_opt = fit_params_mat[0, i, :]
        c = next(color)
        y_fit = abs(my_func(x_fit, *tuple(p_opt))/i_damage_vec[i])
        y_data = abs(bvec[i, :]*2*10**(-7) / (1000))
        #ax.plot(x_data[2:11], -y_data[2:11]/i_damage_vec[i], color=c, marker='x', linestyle='')
        ax.plot(x_data[2:11], y_data[2:11], color=c, marker='x', linestyle='')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=None, useLocale=None, useMathText=None)
        ax.plot(x_fit, y_fit, color=c, linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
        #ax.plot(x_fit, y_fit/p_opt[0], color=c, linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
        ax.set_xlim(-0.55, 0.55)
        ax.set_xlabel(r'$x[m]$', fontsize=20)
        ax.set_ylabel(r'$B_x/I_{tot}$', fontsize=20)
        ax.legend()
        plt.grid(True)

    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(1, 1, 1)
    color2 = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))
    for i in range(n_f):
        c = next(color2)
        x_data = np.array([-1.0, -0.75, -0.5, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.5, 0.75, 1.0])
        p_opt = fit_params_mat[0, i, :]
        y_fit_ideal = abs(my_func(x_data, *tuple(p_opt)))
        y_data_real = abs(bvec[i, :] * i_damage_vec[i] *2*10**(-7) / 1000)
        y_ideal = abs(bvec_i[i, :] * i_ideal_vec[i] *2*10**(-7) / 1000)
        #ax2.plot(x_data, (y_data_real/i_damage_vec[i]-y_fit_ideal/abs(p_opt[0])) / (y_fit_ideal/abs(p_opt[0])), color=c, marker='x', linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
        ax2.plot(x_data, (y_data_real/i_damage_vec[i]-y_fit_ideal/i_ideal_vec[i]) / (y_fit_ideal/i_ideal_vec[i]), color=c, marker='x', linestyle='-', label='f=' + str(freq_list[i]) + "Hz")
        #ax2.plot(x_data, (y_data_real/i_damage_vec[i]-(y_ideal/i_ideal_vec[i])) / (y_ideal/i_ideal_vec[i]), color=c, linestyle='--', label='sim data f=' + str(freq_list[i]) + "Hz")

        y_data_real = bvec[i, :]
        y_ideal = bvec_i[i, :]
        ax2.plot(x_data, (y_data_real-(y_ideal)) / (y_ideal), color=c, linestyle='--', label='sim data f=' + str(freq_list[i]) + "Hz")
        #print(y_ideal)


        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=None, useLocale=None, useMathText=None)
    ax2.set_xlim(-1.0, 1.0)
    ax2.set_xlabel(r'$x[m]$', fontsize=20)
    ax2.set_ylabel(r'$(\tilde{B}^d_x-\tilde{B}^i_x)/\tilde{B}_x^i$', fontsize=20)
    ax2.legend()
    plt.grid(True)
    #    y_fit = my_func(x_fit, *tuple(p_opt))
    #    ax2.plot(x_data, -y_data/p_opt[0], marker='x', linestyle='', label='f=' + str(freq_list[i]) + "Hz")
    #    ax2.plot(x_fit, -y_fit/p_opt[0], linestyle='-', label='f=' + str(freq_list[i]) + "Hz")

    fig3 = plt.figure(figsize=(6, 4))
    color3 = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))
    ax3 = fig3.add_subplot(1, 1, 1)
    for i in range(n_f):
        c = next(color3)
        x_data = np.array([-1.0, -0.75, -0.5, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.5, 0.75, 1.0])
        p_opt = fit_params_mat[0, i, :]

        y_fit_ideal = my_func(x_data, *tuple(p_opt))
        #y_data_real = bvec[i, :] / (1000 * factor)
        y_sim_ideal = bvec_i[i, :] / (1000 * factor)
        #ax3.plot(x_data, y_fit_ideal, color=c, marker='x', linestyle='-', label='fit f=' + str(freq_list[i]) + "Hz")
        ax3.plot(x_data, y_sim_ideal-y_fit_ideal, color=c, linestyle='--', label='sim f=' + str(freq_list[i]) + "Hz")
    ax3.legend()
    plt.grid(True)
    plt.show()

    # print(np.shape(bvec_i))
