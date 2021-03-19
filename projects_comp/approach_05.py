import numpy as np
import code.fit_funcs as ff
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from decimal import Decimal
import math


def notebook_05():
    x0 = 1.0
    y0 = -0.5
    alpha = 0.25 * np.pi
    I = 1.0

    a = 1.0
    b = 0.5

    phi_det = np.linspace(0, 2 * np.pi, 15)
    phi_cont = np.linspace(0, 2 * np.pi, 100)

    ell_det_pts = ff.ellipse_pts(phi_det, x0, y0, alpha, a, b)

    ell_pts_cont = ff.ellipse_pts(phi_cont, x0, y0, alpha, a, b)

    ell_tang_vecs = ff.ellipse_tang_vecs(phi_det, alpha, a, b)
    ell_tang_vecs_cont = ff.ellipse_tang_vecs(phi_cont, alpha, a, b)

    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(ell_det_pts[0], ell_det_pts[1])
    ax1.plot(ell_pts_cont[0], ell_pts_cont[1])
    ax1.quiver(ell_det_pts[0], ell_det_pts[1], ell_tang_vecs[0], ell_tang_vecs[1])
    ax1.set_xlabel(r'x[m]')
    ax1.set_ylabel(r'y[m]')
    ax1.axis('equal')

    ax2 = fig.add_subplot(1, 2, 2)

    ax2.plot(phi_cont / np.pi, ell_tang_vecs_cont[0], label=r'$e_{\varphi,x}$')
    ax2.plot(phi_cont / np.pi, ell_tang_vecs_cont[1], label=r'$e_{\varphi,y}$')
    ax2.set_xlabel(r'$\varphi/\pi$')
    ax2.legend()

    fig = plt.figure(figsize=(6, 3))

    x = ell_det_pts[0]
    y = ell_det_pts[1]
    a_t = a / b
    b_vec = ff.f_b_field_elliptic_pts(x, y, I, a_t, x0, y0, alpha)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(ell_det_pts[0], ell_det_pts[1])
    ax1.plot(ell_pts_cont[0], ell_pts_cont[1])
    ax1.quiver(ell_det_pts[0], ell_det_pts[1], b_vec[0], b_vec[1])
    ax1.set_xlabel(r'x[m]')
    ax1.set_ylabel(r'y[m]')
    ax1.axis('equal')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(phi_cont / np.pi, ell_tang_vecs_cont[0])
    ax2.plot(phi_cont / np.pi, ell_tang_vecs_cont[1])
    ax2.scatter(phi_det / np.pi, b_vec[0]/np.sqrt(b_vec[0]**2 + b_vec[1]**2), label=r'$e_{\varphi,x}$')
    ax2.scatter(phi_det / np.pi, b_vec[1]/np.sqrt(b_vec[0]**2 + b_vec[1]**2), label=r'$e_{\varphi,y}$')
    ax2.set_xlabel(r'$\varphi/\pi$')
    ax2.legend()

    plt.show()
