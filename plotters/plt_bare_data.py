import numpy as np
import matplotlib.pyplot as plt
import code.fit_funcs as ff


def plot_bare_signal(step, volt_list, freq_list, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)
    for i in range(n_f):
        x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
        y_data = np.array(volt_list)[step, :, i, 1] / factor
        ax.plot(x_data, y_data, marker='x', linestyle='', label='f=' + str(freq_list[i]) + "Hz")

    ax.legend()


def plot_bare_signal_and_fit(step, volt_list, freq_list, fit_params_mat, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)

    x_fit = np.linspace(-0.5, 0.5, 10000)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))

    for i in range(n_f):
        c = next(color)
        # bare data
        x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
        y_data = np.array(volt_list)[step, :, i, 1] / factor

        # fit data
        ax.plot(x_data, y_data, marker='x', linestyle='', c=c, label='f=' + str(freq_list[i]) + "Hz")

        p_opt = fit_params_mat[step, i, :]
        y_fit = ff.f_b_field(x_fit, p_opt[0], p_opt[1])
        ax.plot(x_fit, y_fit, linestyle='-', c=c, label='f=' + str(freq_list[i]) + "Hz")

    ax.legend()


def plot_bare_signal_and_fit_norm(step, volt_list, freq_list, fit_params_mat, fit_func, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fig = plt.figure(figsize=(3, 2))
    ax = fig.add_subplot(1, 1, 1)

    x_fit = np.linspace(-0.5, 0.5, 10000)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))

    for i in range(n_f):
        c = next(color)
        p_opt = fit_params_mat[step, i, :]

        # bare data normalized by fitted current
        x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
        y_data = np.array(volt_list)[step, :, i, 1] / factor

        ax.plot(x_data, y_data / p_opt[0], marker='x', linestyle='', c=c, label='f=' + str(freq_list[i]) + "Hz")

        y_fit = fit_func(x_fit, *tuple(p_opt))

        ax.plot(x_fit, y_fit / p_opt[0], linestyle='-', c=c, label='f=' + str(freq_list[i]) + "Hz")

    ax.legend()
