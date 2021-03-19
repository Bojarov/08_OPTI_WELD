import numpy as np
import code.load_and_store as las
import code.fit_funcs_ellipse as ffe
import code.fit_paras as fp
import matplotlib.pyplot as plt
import plotters.plt_comp_data as pcd


def notebook_01():
    freq_list, volt_list = las.load_freq_volt()
    n_steps, n_det, n_f, _ = np.shape(volt_list)
    x_det = np.array([-0.5, -0.16667, 0.16667, 0.5])
    y_det = 2 * np.ones(len(x_det))
    z_det = np.zeros(len(x_det))
    n_steps= 10

    x_data = np.transpose(np.array([x_det, y_det, z_det]))

    alpha_bg_vec = np.zeros(n_steps)
    alpha_ec_vec = np.zeros(n_steps)
    xs_cc_vec = np.zeros(n_steps)
    xs_bg_vec = np.zeros(n_steps)
    x0b_cc_vec = np.zeros(n_steps)
    y0b_cc_vec = np.zeros(n_steps)
    xs_ec_vec = np.zeros(n_steps)
    xs_pre_vec = np.zeros(n_steps)

    #for i in range(10):
    for i in range(n_steps):
        # simple circle fit
        p_opt_pre = fp.fit_params_bx_by(i, 0, ffe.model_function_geometry_circle_all, x_data, volt_list)
        I_pre, x0_pre, y0_pre = p_opt_pre
        xs_pre_vec[i] = x0_pre

    for i in range(n_steps):
    # circle + circle fit
        p0_cc = [I_pre, x0_pre, y0_pre, -40, 2]
        p_opt_cc = fp.fit_params_bx_by(i, 0, ffe.model_function_geometry_circle_b_all, x_data, volt_list, p0_cc)
        I_cc, x0_cc, y0_cc, x0b_cc, y0b_cc = p_opt_cc
        x0b_cc_vec[i] = x0b_cc
        y0b_cc_vec[i] = y0b_cc




    for i in range(n_steps):
        # ellipse + circle
        p0_ec = [I_pre, x0_pre, y0_pre, 0, 0, -40, 2]
        p_opt_ec = fp.fit_params_bx_by(i, 0, ffe.model_function_geometry_ellipse_b_all, x_data, volt_list,
                                       p0=p0_ec)

        I_ec, x0_ec, y0_ec, lamb_ec, alpha_ec, bx0b_ec, by0b_ec = p_opt_ec

        if lamb_ec < 0:
            alpha_ec = alpha_ec - 0.5 * np.pi
        alpha_ec = ((alpha_ec % (2 * np.pi)) / np.pi) % 1

        alpha_ec_vec[i] = alpha_ec
        xs_ec_vec[i] = x0_ec




        # ellipse + background
        #p0_bg = [I_pre, 0, 0, 0, 0, 0, 0]
        #p_opt_bg = fp.fit_params_bx_by(i, 0, ffe.model_function_geometry_ellipse_background_all, x_data, volt_list,
        #                               p0=p0_bg)

        #I_bg, x0_bg, y0_bg, lamb_bg, alpha_bg, bx0b_bg, by0b_bg = p_opt_bg

        #if lamb_bg < 0:
        #    alpha_bg = alpha_bg - 0.5 * np.pi
        #alpha_bg = ((alpha_bg % (2 * np.pi)) / np.pi) % 1

        #alpha_bg_vec[i] = alpha_bg
        #xs_bg_vec[i] = x0_bg

    #plt.plot(xs_pre_vec, linestyle='--', label= 'shift pre')
    #plt.plot(xs_bg_vec, linestyle=':')
    #plt.plot(alpha_bg_vec, linestyle=':')
    #plt.plot(xs_ec_vec)


    fig = plt.figure(figsize=(9, 8))

    ax1 = fig.add_subplot(2, 2, 1)


    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(x0b_cc_vec, label = 'xb cc')
    ax2.plot(y0b_cc_vec, label = 'yb cc')

    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax2.legend()
    plt.show()

def notebook_02():
    """

    """
    recalc = False
    folder_path = './data/data_model_comp/'
    freq_list, volt_list = las.load_freq_volt()
    n_steps, n_det, n_f, _ = np.shape(volt_list)

    x_det = np.array([-0.5, -0.16667, 0.16667, 0.5])
    y_det = 2 * np.ones(len(x_det))
    z_det = np.zeros(len(x_det))
    x_data = np.transpose(np.array([x_det, y_det, z_det]))

    if recalc:
        # create circle fit data
        p_opt_mat_pre = fp.fit_mat_bx_by(ffe.model_function_geometry_circle_all, x_data, volt_list)
        las.save_array(folder_path, 'p_opt_mat_pre', p_opt_mat_pre)

        # p0_ec = [I_pre, x0_pre, y0_pre, 0, 0, -40, 2]
        p0_add_ec = np.array([0, 0, -40, 2])

        # create ellipse + circle fit data
        p_opt_mat_ec = fp.fit_mat_bx_by(ffe.model_function_geometry_ellipse_b_all, x_data, volt_list, p_opt_mat_pre,
                                        p0_add_ec)
        las.save_array(folder_path, 'p_opt_mat_ec', p_opt_mat_ec)

    # ec: I, x0, y0, lamb, alpha, x0b, y0b

    p_opt_mat_pre = las.load_array(folder_path, 'p_opt_mat_pre')
    p_opt_mat_ec = las.load_array(folder_path, 'p_opt_mat_ec')
    # Todo clean up the alphas here!
    # print(np.shape(p_opt_mat_ec))
    for i in range(n_steps):
        for j in range(n_f):
            lamb1 = p_opt_mat_ec[i, j, 3]
            lamb_f = np.arctanh(lamb1)
            p_opt_mat_ec[i, j, 3] = lamb_f

            if lamb_f < 0:
                p_opt_mat_ec[:, :, 4] = p_opt_mat_ec[:, :, 4]-0.5*np.pi

    #            alpha = alpha - 0.5 * np.pi
    #
    p_opt_mat_ec[:, :, 4] = ((p_opt_mat_ec[:, :, 4] % (2 * np.pi)) / np.pi) % 1

    # pcd.opt_plot_pre(p_opt_mat_pre)

    pcd.opt_plot(p_opt_mat_ec)

    plt.show()

    # TODO write plotters for the data above and compare the models 4 models cc, ec, cb, eb
    # Todo most interesting is the ec model!
