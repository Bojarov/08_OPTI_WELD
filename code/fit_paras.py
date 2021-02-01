from inspect import signature
from scipy.optimize import curve_fit
import numpy as np


def fit_params(my_func, volt_list, factor=89000):
    sig = signature(my_func)
    n_steps, _, n_f, _ = np.shape(volt_list)
    n_fit_params = len(list(sig.parameters)) - 1
    print(np.shape(volt_list))

    fit_params_mat = np.zeros((n_steps, n_f, n_fit_params))

    for i in range(n_steps):
        for j in range(n_f):
            x_data = np.array([-0.5, -0.16667, 0.16667, 0.5])
            y_data = np.array(volt_list)[i, :, j, 1] / factor
            p_opt, p_cov = curve_fit(my_func, x_data, y_data)

            fit_params_mat[i, j, :] = p_opt
    return fit_params_mat
