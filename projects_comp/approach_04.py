import numpy as np
import code.fit_funcs as ff
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from decimal import Decimal
import math


def notebook_04():
    # y_det = 0
    # x_det = np.linspace(-1.0, 1.0, 10)
    # x0_i = 0.1
    # y0_i = 1.543
    # I = 1.0
    # a = 0.5
    # b = 1.0
    # a_t = a / b
    # alpha = 0.1 * np.pi
    # print("input parameters")
    # print("I,    x0_i, y0_i,  a_t, alpha/pi")
    # print(np.around(np.array([I, x0_i, y0_i, a_t, alpha / np.pi]), 3))

    # b_pts = ff.f_b_field_elliptic_pts(x_det, y_det, I, a_t, x0_i, y0_i, alpha)
    # b_pts_x = b_pts[0, :]
    # b_pts_y = b_pts[1, :]

    # p_opt_pre, _ = curve_fit(ff.f_b_field_off, x_det, b_pts_x)
    # i_start, y_start, x_start = p_opt_pre
    # print("i_start, x_start, y_start")
    # print(np.around(np.array([i_start, x_start, y_start]), 3))
    # p_opt, _ = curve_fit(ff.f_b_field_elliptic_fit, x_det, b_pts_x,
    #                    bounds=([0.9, 0.4, x_start - 0.1, y_start - 0.2, 0.0 * np.pi],
    #                            [1.1, 1.1, x_start + 0.1, y_start + 0.2, 0.5 * np.pi]))

    # print("fitted parameters")
    # print("I,    x0_i, y0_i,  a_t, alpha/pi")

    # I_f, a_t_f, x0_i_f, y0_i_f, alpha_f = p_opt
    # print(np.around(np.array([I_f, x0_i_f, y0_i_f, a_t_f, alpha_f / np.pi]), 3))

    phi_vec = np.linspace(0, 2 * np.pi, 41)
    phi_vec_cont = np.linspace(0, 2 * np.pi, 400)
    x0 = 1.0
    y0 = -2.4
    a = 1.0
    b = 0.5
    alpha = .0 * np.pi
    ell_pts = ff.ellipse_pts(phi_vec, x0, y0, alpha, a, b)
    ell_pts_cont = ff.ellipse_pts(phi_vec_cont, x0, y0, alpha, a, b)

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(ell_pts[0], ell_pts[1])
    ax1.plot(ell_pts_cont[0], ell_pts_cont[1])
    ax1.axis('equal')
    I = 1.0
    a_t = a / b
    # print("b/a")
    # print(b / a)
    lamb = (a_t - 1) / (a_t + 1)
    u_ell = np.pi * b * (1 + a_t) * (1 + 3 * lamb ** 2 / (10 + np.sqrt(4 - 3 * lamb ** 2)))
    b_pts = ff.f_b_field_elliptic_pts(ell_pts[0], ell_pts[1], I, a_t, x0, y0, alpha)
    b_pts_x = b_pts[0, :]
    b_pts_y = b_pts[1, :]
    mu0 = 4 * np.pi * 10 ** (-7)

    # print("b field norm")
    # print(np.sqrt(b_pts_x ** 2 + b_pts_y ** 2))

    ax1.quiver(-ell_pts[0], ell_pts[1], b_pts_x / mu0 * u_ell, b_pts_y / mu0 * u_ell)
    plt.grid()

    ax2 = fig.add_subplot(1, 2, 2)

    ell_pts_simple = [a * np.cos(alpha) * np.cos(phi_vec) - b * np.sin(alpha) * np.sin(phi_vec),
                      a * np.sin(alpha) * np.cos(phi_vec) + b * np.cos(alpha) * np.sin(phi_vec)]

    norm = np.sqrt((-a * np.cos(alpha) * np.sin(phi_vec) - b * np.sin(alpha) * np.cos(phi_vec)) ** 2
                   + (-a * np.sin(alpha) * np.sin(phi_vec) + b * np.cos(alpha) * np.cos(phi_vec)) ** 2)

    e_phi_simple = np.array([-a * np.cos(alpha) * np.sin(phi_vec) - b * np.sin(alpha) * np.cos(phi_vec),
                             - a * np.sin(alpha) * np.sin(phi_vec) + b * np.cos(alpha) * np.cos(phi_vec)]) / norm

    ax2.scatter(ell_pts_simple[0], ell_pts_simple[1])
    print("phi simple")
    print("y")
    print(ell_pts_simple[1])
    y = ell_pts_simple[1]+y0
    print("x")
    x = ell_pts_simple[0]+x0
    print(ell_pts_simple[0])
    #phi = np.arctan(a_t * (np.cos(alpha) * (y - y0) - np.sin(alpha) * (x - x0)) /
    #                (np.cos(alpha) * (x - x0) + np.sin(alpha) * (y - y0)))
    b_vec = np.sqrt(((y - y0) * np.cos(alpha) - (x - x0) * np.sin(alpha)) ** 2
                + ((y - y0) * np.sin(alpha) + (x - x0) * np.cos(alpha)) ** 2 / a_t ** 2)
    a_vec = a_t * b_vec

    phi = np.arctan2(a_vec * (np.cos(alpha) * (y - y0) - np.sin(alpha) * (x - x0)),
                     b_vec * (np.cos(alpha) * (x - x0) + np.sin(alpha) * (y - y0)))%(2*np.pi)


    ax2.plot(ell_pts_cont[0], ell_pts_cont[1])

    ax2.quiver(ell_pts_simple[0], ell_pts_simple[1], e_phi_simple[0], e_phi_simple[1])

    ax2.axis('equal')
    plt.grid()

    fig = plt.figure(figsize=(6, 3))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(phi, b_pts_x)
    ax1.plot(-(phi-np.pi)%(2*np.pi), b_pts_y)
    norm = np.sqrt(e_phi_simple[0] ** 2 + e_phi_simple[1] ** 2)
    ax1.plot(phi_vec, e_phi_simple[0] / norm, linestyle='--')
    ax1.plot(phi_vec, e_phi_simple[1] / norm, linestyle='--')


    ax2 = fig.add_subplot(1, 2, 2)


    ax2.plot(phi_vec, e_phi_simple[0] / norm)
    ax2.plot(phi_vec, e_phi_simple[1] / norm)

    plt.show()
