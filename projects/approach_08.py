import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import code.fit_funcs_ellipse as ffe


def notebook_08():
    x0 = 0.1
    y0 = 0.1
    alpha = 0.35 * np.pi
    I = 1.0

    a = 1
    b = 0.95
    a_t = a / b
    lamb = (a_t - 1) / (a_t + 1)

    n_det = 10
    i_xyz = 0

    det_pts = np.zeros((n_det, 3))
    det_pts[:, 0] = np.linspace(-0.5, 0.5, n_det)
    det_pts[:, 1] = np.ones(n_det) * 2
    det_pts[:, 2] = np.zeros(n_det)

    phi_ell = np.linspace(0, 2 * np.pi, n_det)
    phi_cont = np.linspace(0, 2 * np.pi, 100)
    ell_pts_cont = ffe.ellipse_pts(phi_cont, x0, y0, alpha, a, b)
    circ_pts_cont = ffe.ellipse_pts(phi_cont, x0, y0, alpha, 1, 1)

    b_x_det = ffe.b_field_ellipse(I, det_pts[:, 0], det_pts[:, 1], x0, y0, lamb, alpha)[:, i_xyz]

    ell_det_pts = ffe.ellipse_pts(phi_ell, x0, y0, alpha, a, b)

    b_ell = ffe.b_field_ellipse(I, ell_det_pts[:, 0], ell_det_pts[:, 1], x0, y0, lamb, alpha)

    x_data = det_pts
    # I, x0, y0, lamb, alpha
    p_opt_ell, _ = curve_fit(ffe.model_function_geometry_ellipse, x_data, b_x_det, p0=[1, 0.1, 0.1, 0, 0.3 * np.pi])

    I_f, x0_f, y0_f, lamb1, alpha_f = p_opt_ell

    lamb_f = np.tan(np.pi * lamb1) / 2

    a_t_1 = (1 + lamb1) / (1 - lamb1)
    a_t_f = (1 + lamb_f) / (1 - lamb_f)

    if lamb_f < 0:
        a_t_1 = 1 / a_t_1
        a_t_f = 1 / a_t_f
        alpha_f = alpha_f - 0.5 * np.pi

    a_f = 1
    b_f = a_f / a_t_1

    ell_fit_pts = ffe.ellipse_pts(phi_cont, x0_f, y0_f, alpha_f, a_f, b_f)

    print("fit params for elliptic fit")
    print(" I,  I_f", I, I_f)
    print("x0, x0_f", x0, x0_f)
    print("y0, y0_f", y0, y0_f)
    print("lamb, lamb1, lamb_f", lamb, lamb1, lamb_f)
    print("a_t, a_t_1, a_t_f", a_t, a_t_1, a_t_f)
    print("alpha, alpha_f", alpha / np.pi, (alpha_f % (2 * np.pi)) / np.pi)

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(det_pts[:, 0], det_pts[:, 1])
    ax1.scatter(ell_det_pts[:, 0], ell_det_pts[:, 1])
    ax1.plot(ell_fit_pts[:, 0], ell_fit_pts[:, 1])
    ax1.quiver(ell_det_pts[:, 0], ell_det_pts[:, 1], b_ell[:, 0], b_ell[:, 1])

    ax1.plot(ell_pts_cont[:, 0], ell_pts_cont[:, 1])
    ax1.plot(circ_pts_cont[:, 0], circ_pts_cont[:, 1], linestyle='--', color='black')

    ax1.scatter(x0, y0, marker='x', label="true")
    ax1.scatter(x0_f, y0_f, marker='x', label="elliptic fit")

    ax1.set_xlabel(r'x[m]')
    ax1.set_ylabel(r'y[m]')
    ax1.axis('equal')
    ax1.legend()

    plt.grid()

    p_opt_circ, _ = curve_fit(ffe.model_function_geometry_circle, x_data, b_x_det)
    I_f_c, x0_f_c, y0_f_c = p_opt_circ
    circ_pts_f = ffe.ellipse_pts(phi_cont, x0_f_c, y0_f_c, 0, 1, 1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(circ_pts_f[:, 0], circ_pts_f[:, 1])
    ax2.scatter(det_pts[:, 0], det_pts[:, 1])
    ax2.scatter(x0, y0, marker='x', label="true")
    ax2.scatter(x0_f_c, y0_f_c, marker='x', label="circular fit")
    ax2.set_xlabel(r'x[m]')
    ax2.set_ylabel(r'y[m]')
    ax2.axis('equal')
    plt.grid()
    ax2.legend()
    plt.show()


def notebook_09():
    x0 = 0.1
    y0 = 0.1
    alpha = 0.35 * np.pi
    I = 1.0

    a = 1
    b = 0.95
    a_t = a / b
    lamb = (a_t - 1) / (a_t + 1)

    n_det = 10
    i_xyz = 0

    det_pts = np.zeros((n_det, 3))
    det_pts[:, 0] = np.linspace(-0.5, 0.5, n_det)
    det_pts[:, 1] = np.ones(n_det) * 2
    det_pts[:, 2] = np.zeros(n_det)

    phi_ell = np.linspace(0, 2 * np.pi, n_det)
    phi_cont = np.linspace(0, 2 * np.pi, 100)
    ell_pts_cont = ffe.ellipse_pts(phi_cont, x0, y0, alpha, a, b)
    circ_pts_cont = ffe.ellipse_pts(phi_cont, x0, y0, alpha, 1, 1)

    b_x_det = ffe.b_field_ellipse(I, det_pts[:, 0], det_pts[:, 1], x0, y0, lamb, alpha)[:, i_xyz]

    ell_det_pts = ffe.ellipse_pts(phi_ell, x0, y0, alpha, a, b)

    b_ell = ffe.b_field_ellipse(I, ell_det_pts[:, 0], ell_det_pts[:, 1], x0, y0, lamb, alpha)

    x_data = det_pts
    # I, x0, y0, lamb, alpha
    p_opt_ell, _ = curve_fit(ffe.model_function_geometry_ellipse, x_data, b_x_det, p0=[1, 0.1, 0.1, 0, 0.3 * np.pi])

    I_f, x0_f, y0_f, lamb1, alpha_f = p_opt_ell

    lamb_f = np.tan(np.pi * lamb1) / 2

    a_t_1 = (1 + lamb1) / (1 - lamb1)
    a_t_f = (1 + lamb_f) / (1 - lamb_f)

    a_f = 1
    b_f = a_f / a_t_1
    ell_fit_pts = ffe.ellipse_pts(phi_cont, x0_f, y0_f, alpha_f, a_f, b_f)

    print(" I,  I_f", I, I_f)
    print("x0, x0_f", x0, x0_f)
    print("y0, y0_f", y0, y0_f)
    print("lamb, lamb1, lamb_f", lamb, lamb1, lamb_f)
    print("a_t, a_t_1, a_t_f", a_t, a_t_1, a_t_f)
    print("alpha, alpha_f", alpha / np.pi, (alpha_f % (2 * np.pi)) / np.pi)

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(det_pts[:, 0], det_pts[:, 1])
    ax1.scatter(ell_det_pts[:, 0], ell_det_pts[:, 1])
    ax1.plot(ell_fit_pts[:, 0], ell_fit_pts[:, 1])
    ax1.quiver(ell_det_pts[:, 0], ell_det_pts[:, 1], b_ell[:, 0], b_ell[:, 1])

    ax1.plot(ell_pts_cont[:, 0], ell_pts_cont[:, 1])
    ax1.plot(circ_pts_cont[:, 0], circ_pts_cont[:, 1], linestyle='--', color='black')

    ax1.scatter(x0, y0)
    ax1.scatter(x0_f, y0_f)

    ax1.set_xlabel(r'x[m]')
    ax1.set_ylabel(r'y[m]')
    ax1.axis('equal')

    plt.grid()

    p_opt_circ, _ = curve_fit(ffe.model_function_geometry_circle, x_data, b_x_det)
    print(p_opt_circ)

    I_f_c, x0_f_c, y0_f_c = p_opt_circ
    circ_pts_f = ffe.ellipse_pts(phi_cont, x0_f_c, y0_f_c, 0, 1, 1)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(circ_pts_f[:, 0], circ_pts_f[:, 1])
    ax2.scatter(det_pts[:, 0], det_pts[:, 1])
    ax2.scatter(x0, y0)
    ax2.scatter(x0_f_c, y0_f_c)
    ax2.set_xlabel(r'x[m]')
    ax2.set_ylabel(r'y[m]')
    ax2.axis('equal')
    plt.grid()

    plt.show()
