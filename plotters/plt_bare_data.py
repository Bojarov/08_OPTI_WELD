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

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)

    x_fit = np.linspace(-0.5, 0.5, 10000)
    mean_shift = np.sum(fit_params_mat[step, :, 2])/n_f

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))

    for i in range(n_f):
        c = next(color)
        p_opt = fit_params_mat[step, i, :]

        # bare data normalized by fitted current
        x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
        y_data = np.array(volt_list)[step, :, i, 1] / factor

        ax.plot(x_data-mean_shift, y_data / p_opt[0], marker='x', linestyle='', c=c, label='f=' + str(freq_list[i]) + "Hz")

        y_fit = fit_func(x_fit, *tuple(p_opt))

        ax.plot(x_fit-mean_shift, y_fit / p_opt[0], linestyle='-', c=c, label='f=' + str(freq_list[i]) + "Hz")
    plt.grid(True)
    ax.legend()


def plot_bare_signal_and_fit_norm_shifted(step, volt_list, freq_list, fit_params_mat, fit_func, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    # bare data normalized by fitted current
    x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])

    # fit date to compare to
    x_fit = np.linspace(-0.5, 0.5, 10000)
    p_opt_base = fit_params_mat[step, 0, :]
    y_fit_base = fit_func(x_fit, *tuple(p_opt_base))

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))

    mean_shift = np.sum(fit_params_mat[step, :, 2])/n_f

    for i in range(n_f):
        c = next(color)
        p_opt = fit_params_mat[step, i, :]

        print(p_opt)
        y_data = np.array(volt_list)[step, :, i, 1] / factor

        ax.plot(x_data-mean_shift, y_data / p_opt[0] , marker='x', linestyle='', c=c,
                label='f=' + str(freq_list[i]) + "Hz")
        y_fit = fit_func(x_fit, *tuple(p_opt))
        #c = next(color)
        ax.plot(x_fit-mean_shift, y_fit / p_opt[0], linestyle='-', c=c, label='fit f=' + str(freq_list[i]) + "Hz")
        #ax.plot(x_fit-mean_shift, (y_data-y_fit)/y_data, marker = 'x', linestyle='-', c=c, label='f=' + str(freq_list[i]) + "Hz")
        #ax.plot(x_fit, (y_data-y_fit)/y_data)
        #ax.plot(x_fit, (y_data / p_opt[0] - y_fit_base / p_opt_base[0]) / (y_fit_base / p_opt_base[0]))
    ax.set_xlim(-0.5, 0.5)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=None, useLocale=None, useMathText=None)
    plt.grid(True)
    ax.legend()


def plot_rel_diff_bare_signal_and_fit_norm_shifted(step, volt_list, freq_list, fit_params_mat, fit_func, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    # bare data normalized by fitted current
    x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])

    # fit date to compare to
    x_fit = x_data  # np.linspace(-0.5, 0.5, 10000)
    p_opt_base = fit_params_mat[step, 0, :]
    y_fit_base = fit_func(x_fit, *tuple(p_opt_base))

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))

    mean_shift = np.sum(fit_params_mat[step, :, 2])/n_f

    for i in range(n_f):
        c = next(color)
        p_opt = fit_params_mat[step, i, :]

        print(p_opt)
        y_data = np.array(volt_list)[step, :, i, 1] / factor

        #ax.plot(x_data, y_data / p_opt[0] , marker='x', linestyle='', c=c,
        #        label='f=' + str(freq_list[i]) + "Hz")
        y_fit = fit_func(x_fit, *tuple(p_opt))
        #c = next(color)
        #ax.plot(x_fit, y_fit / p_opt[0], marker = 'x', linestyle='', c=c, label='fit f=' + str(freq_list[i]) + "Hz")
        ax.plot(x_fit-mean_shift, (y_data-y_fit)/y_data, marker = 'x', linestyle='-', c=c, label='f=' + str(freq_list[i]) + "Hz")
        #ax.plot(x_fit, (y_data-y_fit)/y_data)
        #ax.plot(x_fit, (y_data / p_opt[0] - y_fit_base / p_opt_base[0]) / (y_fit_base / p_opt_base[0]))
    ax.set_xlim(-0.5, 0.5)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=None, useLocale=None, useMathText=None)
    plt.grid(True)
    ax.legend()


def plot_bare_signal_and_fit_norm_shift_along_z(volt_list, freq_list, fit_params_mat, fit_func, factor=89000):
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    # bare data normalized by fitted current
    x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])

    asym_f_z=np.zeros((n_steps, n_f))

    color = iter(plt.cm.rainbow(np.linspace(0, 1, n_f)))


    for step in range(n_steps):


        # fit date to compare to
        x_fit = x_data  # np.linspace(-0.5, 0.5, 10000)
        p_opt_base = fit_params_mat[step, 0, :]
        y_fit_base = fit_func(x_fit, *tuple(p_opt_base))
        mean_shift = np.sum(fit_params_mat[step, :, 2])/n_f

        for i in range(n_f):
            p_opt = fit_params_mat[step, i, :]

            y_data = np.array(volt_list)[step, :, i, 1] / factor
            y_fit = fit_func(x_fit, *tuple(p_opt))

            asym_f_z[step, i] = ((y_data - y_fit) / y_data)[0]


    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)

    for i in range(n_f):
        c = next(color)
        ax.plot(asym_f_z[:, i], linestyle='-', c=c, label='f=' + str(freq_list[i]) + "Hz")

    ax.set_xlim(0, 100)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=None, useLocale=None, useMathText=None)

    ax.legend()
