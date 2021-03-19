import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import code.fit_funcs_ellipse as ffe
import code.field_viso_funcs as fvf
from matplotlib import cm
import matplotlib

plt.style.use('seaborn-white')


def notebook_08():
    x0 = 0.1
    y0 = 0.1
    alpha = 0.25 * np.pi
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
    # p_opt_ell, _ = curve_fit(ffe.model_function_geometry_ellipse_new, x_data, b_x_det, p0=[1, 0.1, 0.1, 0, 0.3 * np.pi])
    p_opt_ell, _ = curve_fit(ffe.model_function_geometry_ellipse_new, x_data, b_x_det, p0=[1, 0.1, 0.1, 0, 1])

    I_f, x0_f, y0_f, lamb1, alpha_f = p_opt_ell

    lamb_f = np.arctan(lamb1)

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
    print("alpha, alpha_f", alpha / np.pi, ((alpha_f % (2 * np.pi)) / np.pi) % 1)

    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(det_pts[:, 0], det_pts[:, 1])
    ax1.scatter(ell_det_pts[:, 0], ell_det_pts[:, 1])
    ax1.plot(ell_fit_pts[:, 0], ell_fit_pts[:, 1], linestyle='--')
    ax1.quiver(ell_det_pts[:, 0], ell_det_pts[:, 1], b_ell[:, 0], b_ell[:, 1])

    ax1.plot(ell_pts_cont[:, 0], ell_pts_cont[:, 1], linestyle=':')
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
    x0 = 0.0
    y0 = 0.0
    alpha = 0.25 * np.pi
    I = 1.0

    a = 1
    b = 0.85
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
    p_opt_ell, _ = curve_fit(ffe.model_function_geometry_ellipse, x_data, b_x_det, p0=[1, 0.0, 0.0, 0, 0.25 * np.pi])

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
    I_f_c, x0_f_c, y0_f_c = p_opt_circ

    # p_opt_ell_stab, _ = curve_fit(ffe.model_function_geometry_ellipse, x_data, b_x_det, p0=[I_f_c, x0_f_c, y0_f_c, 0, 0.25 * np.pi])
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
    # print(p_opt_ell_stab-p_opt_ell)
    plt.show()


def notebook_10():
    x0 = 0.25
    y0 = -0.1
    alpha = 0.35 * np.pi
    I = 1.0
    # I, x0, y0, lamb, alpha
    a = 1.0
    b = 0.8
    a_t = a / b
    lamb = (a_t - 1) / (a_t + 1)

    n_det = 9
    det_pts = np.zeros((n_det, 3))
    det_pts[:, 0] = np.linspace(-0.5, 0.5, n_det)
    # print(det_pts[:,0])
    # exit()
    det_pts[:, 1] = np.ones(n_det) * 2
    det_pts[:, 2] = np.zeros(n_det)
    # I, x0, y0, lamb, alpha
    p0_simple = [1, x0, y0, lamb, alpha]
    p0_all = [1, 1, 1, 1, 1]

    phi_ell = np.linspace(0, 2 * np.pi, n_det)
    phi_cont = np.linspace(0, 2 * np.pi, 100)
    ell_pts_cont = ffe.ellipse_pts(phi_cont, x0, y0, alpha, a, b)
    circ_pts_cont = ffe.ellipse_pts(phi_cont, x0, y0, alpha, 1, 1)

    b_det_x = ffe.b_field_ellipse(I, det_pts[:, 0], det_pts[:, 1], x0, y0, lamb, alpha)[:, 0]
    b_det_y = ffe.b_field_ellipse(I, det_pts[:, 0], det_pts[:, 1], x0, y0, lamb, alpha)[:, 1]
    b_det_all = np.concatenate((b_det_x, b_det_y), axis=None)

    ell_det_pts = ffe.ellipse_pts(phi_ell, x0, y0, alpha, a, b)
    b_ell = ffe.b_field_ellipse(I, ell_det_pts[:, 0], ell_det_pts[:, 1], x0, y0, lamb, alpha)

    x_data = det_pts
    # I, x0, y0, lamb, alpha
    p_opt_ell, _ = curve_fit(ffe.model_function_geometry_ellipse, x_data, b_det_x, p0=p0_simple)
    # p_opt_ell, _ = curve_fit(ffe.model_function_geometry_ellipse, x_data, b_det_x)

    I_f, x0_f, y0_f, lamb1, alpha_f = p_opt_ell

    lamb_f = np.tan(np.pi * lamb1) / 2
    a_t_1 = (1 + lamb1) / (1 - lamb1)
    a_t_f = (1 + lamb_f) / (1 - lamb_f)

    a_f = 1
    b_f = a_f / a_t_1
    b_f_1 = a_f / a_t_f
    ell_fit_pts = ffe.ellipse_pts(phi_cont, x0_f, y0_f, alpha_f, a_f, b_f_1)

    print(" I,  I_f", I, I_f)
    print("x0, x0_f", x0, x0_f)
    print("y0, y0_f", y0, y0_f)
    print("lamb, lamb1, lamb_f", lamb, lamb1, lamb_f)
    print("a_t, a_t_1, a_t_f", a_t, a_t_1, a_t_f)
    print("alpha, alpha_f", alpha / np.pi, (alpha_f % (2 * np.pi)) / np.pi)

    # Plotting
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

    p_opt_ell_all, _ = curve_fit(ffe.model_function_geometry_ellipse_all, x_data, b_det_all)

    I_f, x0_f, y0_f, lamb1, alpha_f = p_opt_ell_all

    a_f = 1
    b_f = a_f / a_t_1

    lamb_f = np.tan(np.pi * lamb1) / 2
    a_t_1 = (1 + lamb1) / (1 - lamb1)
    a_t_f = (1 + lamb_f) / (1 - lamb_f)

    if lamb1 < 0:
        print("yes")
        a_t_1 = 1 / a_t_1
        a_t_f = 1 / a_t_f
        alpha_f = alpha_f - 0.5 * np.pi

    a_f = 1
    b_f = a_f / a_t_1

    # ell_fit_pts2 = ffe.ellipse_pts(phi_cont, x0_f, y0_f, alpha_f, a_f, b_f)

    print("all values")
    print(" I,  I_f", I, I_f)
    print("x0, x0_f", x0, x0_f)
    print("y0, y0_f", y0, y0_f)
    print("lamb, lamb1, lamb_f", lamb, lamb1, lamb_f)
    print("a_t, a_t_1, a_t_f", a_t, a_t_1, a_t_f)
    print("alpha, alpha_f", alpha / np.pi, ((alpha_f % (2 * np.pi)) / np.pi) % 1)

    ax2 = fig.add_subplot(1, 2, 2)

    ax2.scatter(det_pts[:, 0], det_pts[:, 1])
    ax2.scatter(x0, y0)
    ax2.set_xlabel(r'x[m]')
    ax2.set_ylabel(r'y[m]')
    ax2.axis('equal')
    plt.grid()

    plt.show()


# TODO regarding notebook 10:
# TODO check if weldline angle can be recovered from: - real data
#                                                    - FH data
# TODO Question: fitting all components stabilizes the curve fit but only in 1st quadrant?

def notebook_10_viso():
    mu0 = 4 * np.pi * 10 ** (-7)
    x0_ell = 0.0
    y0_ell = 0.0
    x0_c = 5.0
    y0_c = 0.0

    alpha = 0.00 * np.pi
    I = 1.0
    # I, x0, y0, lamb, alpha
    a = 1.0
    b = 0.8
    a_t = a / b
    lamb = (a_t - 1) / (a_t + 1)

    extent = (-3, 4, -4, 3)
    x = np.linspace(-1, 1, 400)
    y = np.linspace(-1, 1, 400)

    X, Y = np.meshgrid(x, y)

    Z_e = fvf.b_field_ellipse_mag_2D(I, X, Y, x0_ell, y0_ell, lamb, alpha)
    Z_c = fvf.b_field_mag_2D(-I, X, Y, x0_c, y0_c)
    Z = Z_e  # +Z_c

    levels = 1 / np.linspace(1 / np.min(Z), np.max(Z), 20)

    norm = cm.colors.Normalize(vmax=abs(Z).max(), vmin=Z.min())
    cmap = cm.PRGn

    fig, axs0 = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.3)

    cset1 = axs0.contourf(X, Y, Z, levels, norm=matplotlib.colors.LogNorm(),
                          cmap=cm.get_cmap(cmap, len(levels) - 1))

    cset2 = axs0.contour(X, Y, Z, cset1.levels, colors='k')

    for c in cset2.collections:
        c.set_linestyle('solid')

    # axs0.set_title('Filled contours')
    fig.colorbar(cset1, ax=axs0)

    fig.tight_layout()
    plt.show()


def notebook_11():
    # source specifications
    x0 = -0.25
    y0 = - 0.1
    alpha = 0.25 * np.pi
    I = 1.0
    # I, x0, y0, lamb, alpha
    a = 1.0
    b = 0.9
    a_t = a / b
    lamb = (a_t - 1) / (a_t + 1)

    # returning cable position
    x0b = 45
    y0b = 2.0

    # detector positions
    n_det = 4
    det_pts = np.zeros((n_det, 3))
    det_pts[:, 0] = np.linspace(-0.5, 0.5, n_det)
    det_pts[:, 1] = np.ones(n_det) * 2
    det_pts[:, 2] = np.zeros(n_det)

    del_noise = 10 ** (-9)
    noise_x = (np.random.rand(n_det) - 0.5) * del_noise
    noise_y = (np.random.rand(n_det) - 0.5) * del_noise

    phi_ell = np.linspace(0, 2 * np.pi, n_det)
    phi_cont = np.linspace(0, 2 * np.pi, 100)
    ell_pts_cont = ffe.ellipse_pts(phi_cont, x0, y0, alpha, a, b)
    circ_pts_cont = ffe.ellipse_pts(phi_cont, x0, y0, alpha, 1, 1)

    b_det_x = ffe.b_field_ellipse(I, det_pts[:, 0], det_pts[:, 1], x0, y0, lamb, alpha)[:, 0]
    b_det_y = ffe.b_field_ellipse(I, det_pts[:, 0], det_pts[:, 1], x0, y0, lamb, alpha)[:, 1]
    b_det_all = np.concatenate((b_det_x, b_det_y), axis=None)

    ell_det_pts = ffe.ellipse_pts(phi_ell, x0, y0, alpha, a, b)
    b_ell = ffe.b_field_ellipse(I, ell_det_pts[:, 0], ell_det_pts[:, 1], x0, y0, lamb, alpha)

    x_data = det_pts

    # I, x0, y0, lamb, alpha
    p_opt_ell_all, _ = curve_fit(ffe.model_function_geometry_ellipse_all, x_data, b_det_all)

    I_f, x0_f2, y0_f2, lamb1, alpha_f = p_opt_ell_all

    lamb_f = np.arctan(lamb1)
    a_t_f = (1 + lamb_f) / (1 - lamb_f)

    if lamb1 < 0:
        a_t_f = 1 / a_t_f
        alpha_f = alpha_f - 0.5 * np.pi

    a_f = 1
    b_f = a_f / a_t_f

    ell_fit_pts2 = ffe.ellipse_pts(phi_cont, x0_f2, y0_f2, alpha_f, a_f, b_f)

    print("all values")
    print(" I,  I_f", I, I_f)
    print("x0, x0_f", x0, x0_f2)
    print("y0, y0_f", y0, y0_f2)
    print("lamb, lamb_f", lamb, lamb_f)
    print("a_t, a_t_1, a_t_f", a_t, a_t_f)
    print("alpha, alpha_f", alpha / np.pi, ((alpha_f % (2 * np.pi)) / np.pi) % 1)

    # Plotting
    fig = plt.figure(figsize=(9, 4))

    ax1 = fig.add_subplot(1, 2, 1)

    ax1.plot(ell_pts_cont[:, 0], ell_pts_cont[:, 1], color='b')
    ax1.plot(circ_pts_cont[:, 0], circ_pts_cont[:, 1], linestyle='--', color='black')
    ax1.scatter(det_pts[:, 0], det_pts[:, 1])
    ax1.scatter(ell_det_pts[:, 0], ell_det_pts[:, 1], color='b')
    ax1.plot(ell_fit_pts2[:, 0], ell_fit_pts2[:, 1], linestyle='--', color='orange')
    ax1.quiver(ell_det_pts[:, 0], ell_det_pts[:, 1], b_ell[:, 0], b_ell[:, 1])

    ax1.scatter(x0, y0, color='black')
    ax1.scatter(x0_f2, y0_f2, color='orange', marker='x')

    ax1.set_xlabel(r'x[m]')
    ax1.set_ylabel(r'y[m]')
    ax1.axis('equal')

    plt.grid()

    b_det_x_b = (ffe.b_field_ellipse(I, det_pts[:, 0], det_pts[:, 1], x0, y0, lamb, alpha)[:, 0] +
                 ffe.b_field_circle(-I, det_pts[:, 0], det_pts[:, 1], x0b, y0b)[:, 0]) + noise_x

    b_det_y_b = (ffe.b_field_ellipse(I, det_pts[:, 0], det_pts[:, 1], x0, y0, lamb, alpha)[:, 1] +
                 ffe.b_field_circle(-I, det_pts[:, 0], det_pts[:, 1], x0b, y0b)[:, 1])

    b_det_b_all = np.concatenate((b_det_x_b, b_det_y_b), axis=None)

    # NAIVE CIRCLE
    p_opt_c, _ = curve_fit(ffe.model_function_geometry_circle, x_data, b_det_x_b)
    I_c, x0_c, y0_c = p_opt_c
    circ_pts_naive = ffe.ellipse_pts(phi_cont, x0_c, y0_c, 0, 1, 1)

    # NAIVE CIRCLE ALL COMPS
    p_opt_ca, _ = curve_fit(ffe.model_function_geometry_circle_all, x_data, b_det_b_all)
    I_ca, x0_ca, y0_ca = p_opt_ca
    circ_pts_naive_all = ffe.ellipse_pts(phi_cont, x0_ca, y0_ca, 0, 1, 1)

    # ELLIPTIC FIT WITH BACK CURRENT

    p0_back = [5, 0.1, -1, 0.1, 0.25 * np.pi, 30, 2]
    p_opt_ell_all, _ = curve_fit(ffe.model_function_geometry_ellipse_b_all, x_data, b_det_b_all, p0_back)

    I_fs, x0_fs, y0_fs, lamb1_fs, alpha_fs, x0b_fs, y0b_fs = p_opt_ell_all

    lamb_fs = np.arctanh(lamb1_fs)
    a_t_fs = (1 + lamb_fs) / (1 - lamb_fs)

    if lamb_fs < 0:
        a_t_fs = 1 / a_t_fs
        alpha_fs = alpha_fs - 0.5 * np.pi

    a_fs = 1
    b_fs = a_f / a_t_fs

    ell_fit_pts_b = ffe.ellipse_pts(phi_cont, x0_fs, y0_fs, alpha_fs, a_fs, b_fs)

    print("all values back")
    print(" I,  I_f", I, I_fs)
    print("x0, x0_f", x0, x0_fs)
    print("y0, y0_f", y0, y0_fs)
    print("lamb, lamb_f", lamb, lamb_fs)
    print("a_t, a_t_1, a_t_f", a_t, a_t_f)
    print("alpha, alpha_f", (alpha / np.pi) % 1, ((alpha_fs % (2 * np.pi)) / np.pi) % 1)
    print("x0b, x0_f", x0b, x0b_fs)
    print("y0, y0_f", y0b, y0b_fs)

    # ELLIPTIC FIT WITH CONSTANT BACKGROUND
    p0_bgf = [1, 1, 1, 0.1, 0.25 * np.pi, 0, 0]
    p_opt_ell_background_all, _ = curve_fit(ffe.model_function_geometry_ellipse_background_all, x_data, b_det_b_all,
                                            p0_bgf)

    I_bgf, x0_bgf, y0bgf, lamb1_bgf, alpha_bgf, bx0, by0 = p_opt_ell_background_all
    lamb_bgf = np.arctanh(lamb1_bgf)
    a_t_f = (1 + lamb_bgf) / (1 - lamb_bgf)
    if lamb1_bgf < 0:
        a_t_f = 1 / a_t_f
        alpha_bgf = alpha_bgf - 0.5 * np.pi

    a_f = 1
    b_f = a_f / a_t_f

    ell_fit_pts_bg = ffe.ellipse_pts(phi_cont, x0_bgf, y0bgf, alpha_bgf, a_f, b_f)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(det_pts[:, 0], det_pts[:, 1])
    ax2.plot(circ_pts_cont[:, 0], circ_pts_cont[:, 1], linestyle='--', color='black')
    ax2.plot(circ_pts_naive[:, 0], circ_pts_naive[:, 1], linestyle='-', color='c', label = 'circle x')
    ax2.plot(circ_pts_naive_all[:, 0], circ_pts_naive_all[:, 1], linestyle='--', color='c', label = 'circle x, y')
    #    ax2.plot(circ_fit_pts[:, 0], circ_fit_pts[:, 1], linestyle='-', color='black', label='circle')
    #    ax2.plot(ell_fit_pts_s[:, 0], ell_fit_pts_s[:, 1], color='blue', label='ellipse')
    ax2.plot(ell_pts_cont[:, 0], ell_pts_cont[:, 1], color='orange')
    #    ax2.plot(cc_fit_pts[:, 0], cc_fit_pts[:, 1], linestyle='-', color='orange', label='circle + circle')
    #    ax2.plot(c_bg_pts[:, 0], c_bg_pts[:, 1], linestyle='-', color='cyan', label='circle + background')
    ax2.plot(ell_fit_pts_b[:, 0], ell_fit_pts_b[:, 1], linestyle='--', color='red', label='ellipse + circle')
    ax2.scatter(x0_fs, y0_fs, color='red', marker='x')
    ax2.plot(ell_fit_pts_bg[:, 0], ell_fit_pts_bg[:, 1], color='green', linestyle=':', label='ellipse + background')
    ax2.scatter(x0_bgf, y0bgf, color='green', marker='x')
    #    # ax2.scatter(x0b_f, y0b_f)
    #    ax2.scatter(x0_fs, y0_fs, marker='x', color='blue')
    #    ax2.scatter(x0cs, y0cs, marker='x', color='orange')

    #    ax2.scatter(x0cbg, y0cbg, color='cyan', marker='x')
    #
    #    ax2.scatter(x0circ, y0_circ, marker='x', color='black')
    #    ax2.set_xlabel(r'x[m]')
    #    ax2.set_ylabel(r'y[m]')
    #    # ax2.set_xlim(-2.0,12)
    ax2.axis('equal')
    plt.grid()
    plt.show()
# TODO this notebook 11 compares the different models
#TODO use circle fit with all comps (and maybe background) to get center, then use center for ellipse fit
