import code.load_and_store as las
import code.fit_funcs as ff
import code.fit_paras as fp
import code.chi_test as ct
import plotters.plt_bare_data as pbd
import plotters.plt_bare_sym as pbs
import plotters.plt_fit_para as pfp
import numpy as np




def notebook_00():

    freq_list, volt_list = las.load_freq_volt()

    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fit_params_mat = fp.fit_params(ff.f_b_field, volt_list)

    fit_params_mat_s = fp.fit_params(ff.f_b_field_off, volt_list)

    # pbd.plot_bare_signal(0, volt_list, freq_list)

    # for i in range(10):
    #    #pbd.(i, volt_list, freq_list, fit_params_mat, ff.f_b_field)


    # pbd.plot_bare_signal_and_fit_norm(0, volt_list, freq_list, fit_params_mat, ff.f_b_field)
    # pbd.plot_bare_signal_and_fit_norm(0, volt_list, freq_list, fit_params_mat_s, ff.f_b_field_off)


    # THESE ARE THE LATEST STATEMENTS ON THE SYMMETRY

    # example of "corrected" data, b-field of ideal pipe centered at x = 0, and corresponding data
    pbd.plot_bare_signal_and_fit_norm_shifted(0, volt_list, freq_list, fit_params_mat_s, ff.f_b_field_off)

    # example of "corrected" data, b-field of ideal pipe centered at x = 0, and corresponding data
    #pbd.plot_rel_diff_bare_signal_and_fit_norm_shifted(0, volt_list, freq_list, fit_params_mat_s, ff.f_b_field_off)

    # cleaned relative asymmetry between data and ideal case for one detector point and different positions
    # along the pipeline axis
    # pbd.plot_bare_signal_and_fit_norm_shift_along_z(volt_list, freq_list, fit_params_mat_s, ff.f_b_field_off)
    # comparison between real data asymmetry and shift of detector to pipeline center
    pfp.plot_fit_sym_comp(volt_list, fit_params_mat, fit_params_mat_s, freq_list)

    # pfp.plot_fit_params_f_z(fit_params_mat_s, freq_list, para_ind=2)
    # pfp.plot_dist_fit_params(fit_params_mat_s, freq_list, para_ind=2)
    # pbs.plot_bare_signal_symmetry_f_z(volt_list, freq_list, n_win=10)


    # pbs.plot_bare_signal_symmetry_norm_f_z(volt_list, freq_list, fit_params_mat, n_win=20)

    # pfp.plot_fit_params_f_z(fit_params_mat, freq_list, 0)  # current
    # pfp.plot_fit_params_f_z(fit_params_mat, freq_list, 1)  # depth

    # pfp.plot_fit_params_f_z(fit_params_mat_s, freq_list, 0)  # current
    # pfp.plot_fit_params_f_z(fit_params_mat_s, freq_list, 1)  # depth
    # pfp.plot_fit_params_f_z(fit_params_mat_s, freq_list, 2)  # shift

    # ct.chi_sqrt_data_point(0, 0, volt_list, fit_params_mat, ff.f_b_field)
    # chi_sqrt_mat = ct.chi_sqrt_mat_calc(volt_list, fit_params_mat, ff.f_b_field)
    # chi_sqrt_mat_s = ct.chi_sqrt_mat_calc(volt_list, fit_params_mat_s, ff.f_b_field_off)
    # print(np.shape(chi_sqrt_mat))
    # print(np.shape(fit_params_mat))
    # print(np.shape(chi_sqrt_mat_s))
    # pfp.plot_fit_params_f_z(chi_sqrt_mat, freq_list)  # shift
    # pfp.plot_fit_params_f_z(chi_sqrt_mat_s, freq_list)  # shift