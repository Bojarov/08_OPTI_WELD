import code.load_and_store as las
import numpy as np
import plotters.plt_bare_data as pbd
import plotters.plt_bare_sym as pbs
import scipy.stats as stats
import code.data_sym as ds
import code.fit_paras as fp
import code.fit_funcs as ff
import plotters.plt_fit_para as pfp


def notebook_01():
    freq_list, volt_list = las.load_freq_volt()

    n_steps, n_det, n_f, _ = np.shape(volt_list)

    #y_sym_mat_o = ds.by_sym_mat(volt_list, det_ind=0)
    #y_sym_mat_i = ds.by_sym_mat(volt_list, det_ind=1)

    # print(np.shape(y_sym_mat_o))
    # print(np.shape(y_sym_mat_i))
    # (mu_o, sigma_o) = stats.norm.fit(y_sym_mat_o[:,0])
    # (mu_i, sigma_i) = stats.norm.fit(y_sym_mat_i[:,0])
    # print(mu_o, sigma_o)
    # print(mu_i, sigma_i)
    # print(mu_o*89000, mu_i*89000.0, -mu_i*89000.0, -mu_o*89000.0)

    volt_list_sym = ds.volt_list_sym_calc(volt_list)

    fit_params_mat = fp.fit_params(ff.f_b_field, volt_list_sym)

    fit_params_mat_s = fp.fit_params(ff.f_b_field_off, volt_list_sym)

    pbd.plot_bare_signal_and_fit_norm_shifted(0, volt_list_sym, freq_list, fit_params_mat_s, ff.f_b_field_off)

    # pfp.plot_fit_sym_comp(volt_list_sym, fit_params_mat, fit_params_mat_s, freq_list)

    # pbd.plot_rel_diff_bare_signal_and_fit_norm_shifted(0, volt_list_sym, freq_list, fit_params_mat_s, ff.f_b_field_off)
