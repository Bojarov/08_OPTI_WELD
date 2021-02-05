import numpy as np


def by_sym_mat(volt_list, det_ind, factor=89000):
    n_det = np.shape(volt_list)[1]
    # print(n_det)
    y_sym_mat = (np.array(volt_list)[:, det_ind, :, 1] - np.array(volt_list)[:, n_det - 1 - det_ind, :, 1]) / factor
    return y_sym_mat

def volt_list_sym_calc(volt_list):
    """
    symmetrization between voltages of opposite detectors (e.g. positions on detector arrays are x and -x )
    pairs of detector voltages are shifted to subtract average background asymmetry between both, for example
    due too the conductance line that closes the measurement loop
    average is taken over all points of the data set
    """
    volt_arr = np.array(volt_list)

    n_step, n_det, n_f, _ = np.shape(volt_arr)

    volt_sym = volt_arr[:, :, :, 1] - volt_arr[:, ::-1, :, 1]
    volt_sym_mu = 0.5 * np.sum(volt_sym, axis=0) / n_step
    volt_arr_sym = volt_arr[:, :, :, 1] - volt_sym_mu
    volt_arr[:, :, :, 1] = volt_arr_sym

    return list(volt_arr)
