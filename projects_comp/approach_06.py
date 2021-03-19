import numpy as np
import code.fit_funcs as ff
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from decimal import Decimal
import math


def notebook_06():
    y_det = 0
    x_det = np.linspace(-0.5, 0.5, 10)
    x_det_cont = np.linspace(-0.5, 0.5, 50)
    x0_i = 0.0
    y0_i = 1.0
    I = 1.0
    a = 1.0
    b = 0.8
    a_t = a / b
    alpha = 0.2 * np.pi
    print("input parameters")
    print("I,    x0_i, y0_i,  a_t, alpha/pi")
    print(np.around(np.array([I, x0_i, y0_i, a_t, alpha / np.pi]), 3))

    b_pts = ff.f_b_field_elliptic_field_pts(x_det, y_det, I, a_t, x0_i, y0_i, alpha)

    fig = plt.figure(figsize=(6, 5))

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    a_t_vec = np.linspace(1, 0.9, 5)

    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(a_t_vec))))

    opt_para_circ = []
    opt_para_ell = []

    for a_t in a_t_vec:
        b_pts_cont = ff.f_b_field_elliptic_field_pts(x_det_cont, y_det, I, a_t, x0_i, y0_i, alpha)
        bx_pts_det = ff.f_b_field_elliptic_field_pts(x_det, y_det, I, a_t, x0_i, y0_i, alpha)
        c = next(color)
        ax1.plot(x_det_cont, b_pts_cont[0], color=c, label=r'$a/b=$' + str(a_t))

        p_opt_pre, _ = curve_fit(ff.f_b_field_off, x_det, bx_pts_det[0, :])
        i_pre, y_pre, x_pre = p_opt_pre
        opt_para_circ.append(np.array([i_pre, y_pre, x_pre]))

        bx_circ_f = ff.f_b_field_off(x_det, i_pre, y_pre, x_pre)

        p_opt_ell, _ = curve_fit(ff.f_b_field_elliptic_fit_x, x_det, bx_pts_det[0, :],
                                 bounds=([i_pre - 0.1, 0.8, x_pre - 0.05, y_pre - 0.1, 0.0 * np.pi],
                                         [i_pre + 0.1, 1.2, x_pre + 0.05, y_pre + 0.1, 0.5 * np.pi]))
        i_ell_f, a_t_f, x0_i_f, y0_i_f, alpha_f = p_opt_ell

        opt_para_ell.append(np.array([i_ell_f, a_t_f, x0_i_f, y0_i_f, alpha_f]))

        bx_ell_f = ff.f_b_field_elliptic_fit_x(x_det, i_ell_f, a_t_f, x0_i_f, y0_i_f, alpha_f)

        ax1.plot(x_det, bx_circ_f, linestyle='', marker='x', color=c)  # , label=r'$a/b=$' + str(a_t))
        ax2.plot(x_det, (bx_pts_det[0, :] - bx_circ_f) / bx_pts_det[0, :], linestyle='-', marker='x',
                 color=c)

        ax3.plot(x_det_cont, b_pts_cont[0], color=c, label=r'$a/b=$' + str(a_t))
        ax3.plot(x_det, bx_ell_f, linestyle='', marker='x', color=c)

        ax4.plot(x_det, (bx_pts_det[0, :] - bx_ell_f) / bx_pts_det[0, :], linestyle='-', marker='x',
                 color=c)  # , label=r'$a/b=$' + str(a_t))

    ax1.set_xlabel(r'$x_{det}[m]$')
    ax1.set_ylabel(r'$B_x$')
    ax1.legend()

    ell_para_mat = np.array(opt_para_ell)
    circ_para_mat = np.array(opt_para_circ)

    #print(circ_para_mat)
    #exit()

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(1, 2, 1)

    ax1.plot(a_t_vec, (circ_para_mat[:, 0] - I) / I,  label=r'$\Delta I$')
    ax1.plot(a_t_vec, (circ_para_mat[:, 2] - x0_i), label=r'$y_{0,fit}-y_0$')
    ax1.plot(a_t_vec, (circ_para_mat[:, 1] - y0_i), label=r'$x_{0,fit}-x_0$')

    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)

    ax2.plot(a_t_vec, (ell_para_mat[:, 0] - I) / I,  label=r'$\Delta I$')
    ax2.plot(a_t_vec, (ell_para_mat[:, 2] - x0_i), label=r'$x_{0,fit}-x_0$')
    ax2.plot(a_t_vec, (ell_para_mat[:, 3] - y0_i), label=r'$y_{0,fit}-y_0$')
    ax2.plot(a_t_vec, (ell_para_mat[:, 1] - a_t_vec) / a_t_vec, label=r'$\Delta (a/b)$')
    ax2.plot(a_t_vec, (ell_para_mat[:, 4] - alpha) / alpha, label=r'$\Delta \alpha$')




    ax2.legend()

    # ax2 = fig.add_subplot(1, 2, 1)

    ell_para_mat = np.array(opt_para_ell)
    circ_para_mat = np.array(opt_para_circ)
    # print(np.array(opt_para_ell)[0,0])

    plt.show()
