import code.load_and_store as las
import numpy as np
import plotters.plt_bare_data as pbd
import code.fit_funcs as ff
import code.fit_paras as fp
import plotters.plt_fit_para as pfp


def notebook_01():
    field_data_path = './data_app3/bvec.npy'
    ideal_field_data_path = './data_app3/bvec_i.npy'

    field_data_path_comp = './data_app3/bvec_comp.npy'
    ideal_field_data_path_comp = './data_app3/bvec_i_comp.npy'

    freq_list = np.array([2, 4, 8, 16, 32, 64, 128])
    freq_list_comp = np.array([3, 6, 12, 24, 48, 92])
    x_data = np.array([-0.5, -0.16667, 0.0, 0.16667, 0.5])
    phi_c_arr = np.array([0, 0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi, np.pi])
    mu_0 = 2 * 10 ** (-7)

    # load the normalized fields (b_field/current)
    b_x_phi_f = las.load_array(field_data_path)
    b_x_phi_f_i = las.load_array(ideal_field_data_path)

    # load the normalized fields with same freqs as the real data (b_field/current)
    b_x_phi_f_comp = las.load_array(field_data_path)
    b_x_phi_f_i_comp = las.load_array(ideal_field_data_path)

    b_x_phi_f_trans = abs(np.repeat(b_x_phi_f[:, :, :, np.newaxis], 3, axis=3))
    # b_x_phi_f_i_trans = abs(np.repeat(b_x_phi_f_i[:, :, :, np.newaxis], 3, axis=3))

    b_x_phi_f_trans_comp = abs(np.repeat(b_x_phi_f_comp[:, :, :, np.newaxis], 3, axis=3))
    # b_x_phi_f_i_trans_comp = abs(np.repeat(b_x_phi_f_i_comp[:, :, :, np.newaxis], 3, axis=3))

    #    b_x_phi_f_i = las.load_array(ideal_field_data_path)

    # n_phi, n_det, n_f = np.shape(b_x_phi_f)

    # pbd.plot_bare_signal(0, x_data, b_x_phi_f_i_trans, freq_list, factor=1000)

    # pbd.plot_bare_signal(0, x_data, b_x_phi_f_trans, freq_list, factor=1000)

    fit_params_mat = fp.fit_params(ff.f_b_field_off, x_data, b_x_phi_f_trans, factor=1000 / mu_0)
    fit_params_mat_comp = fp.fit_params(ff.f_b_field_off, x_data, b_x_phi_f_trans_comp, factor=1000 / mu_0)
    #print(np.shape(fit_params_mat_comp))

    #exit()
    # fit_params_mat_i = fp.fit_params(ff.f_b_field_off, b_x_phi_f_i_trans, factor=1000)

    # pbd.plot_bare_signal_and_fit(0, b_x_phi_f_trans, freq_list, fit_params_mat, factor=1000)

    pfp.plot_fit_params_f_step_axis(fit_params_mat, freq_list, para_ind=2)
    pfp.plot_fit_params_f_step_axis_2(fit_params_mat, freq_list, para_ind=2)
    # pfp.plot_fit_params_f_axis(fit_params_mat, phi_c_arr, freq_list, para_ind=2)

    # pfp.plot_fit_params_f_step_axis(fit_params_mat_comp, freq_list_comp, para_ind=2)
    # pfp.plot_fit_params_f_axis(fit_params_mat_comp, phi_c_arr, freq_list_comp, para_ind=2)

    # print(n_phi, n_det, n_f)
