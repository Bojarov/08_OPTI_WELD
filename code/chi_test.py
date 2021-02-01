from scipy.stats import chisquare
import numpy as np


# import matplotlib.pyplot as plt


def chi_sqrt_data_point(i_step, i_f, volt_list, fit_params_mat, fit_func, factor=89000):
    n_steps, _, n_f, _ = np.shape(volt_list)
    x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
    y_data = np.array(volt_list)[i_step, :, i_f, 1] / factor
    p_opt = tuple(fit_params_mat[i_step, i_f, :])
    y_fit = fit_func(x_data, *p_opt)
    # print(np.sum((y_fit-y_data)**2/y_fit))
    return chisquare(y_data, y_fit)


def chi_sqrt_mat_calc(volt_list, fit_params_mat, fit_func):
    n_steps, _, n_f, _ = np.shape(volt_list)

    chi_sqrt_mat = np.zeros((n_steps, n_f, 1))
    for i in range(n_steps):
        for j in range(n_f):
            chi_sqrt_mat[i, j, 0] = chi_sqrt_data_point(i, j, volt_list, fit_params_mat, fit_func)[1]
            a = chi_sqrt_data_point(i, j, volt_list, fit_params_mat, fit_func)[1]
            print(a)

    return chi_sqrt_mat
