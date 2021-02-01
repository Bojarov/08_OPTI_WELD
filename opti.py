import numpy as np
import matplotlib.pyplot as plt
import code.load_and_store as las
import code.fit_funcs as ff
import code.fit_paras as fp
import plotters.plt_bare_data as pbd
import plotters.plt_bare_sym as pbs
import plotters.plt_fit_para as pfp



def main():
    freq_list, volt_list = las.load_freq_volt()

    n_steps, n_det, n_f, _ = np.shape(volt_list)

    fit_params_mat = fp.fit_params(ff.f_b_field, volt_list)

    #pbd.plot_bare_signal(0, volt_list, freq_list)

    pbd.plot_bare_signal_and_fit(0, volt_list, freq_list, fit_params_mat)

    pbs.plot_bare_signal_symmetry_f_z(volt_list, freq_list, n_win=20)

    pbs.plot_bare_signal_symmetry_norm_f_z(volt_list, freq_list, fit_params_mat, n_win=20)

    pfp.plot_depth_f_z(fit_params_mat, freq_list)

    pfp.plot_current_f_z(fit_params_mat, freq_list)

    plt.show()


if __name__ == '__main__':
    main()
