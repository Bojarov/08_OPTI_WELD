import numpy as np


def ellipse_pts(phi, x0, y0, alpha, a, b):
    pts = np.zeros((len(phi), 2))
    pts[:, 0] = x0 + a * np.cos(alpha) * np.cos(phi) - b * np.sin(alpha) * np.sin(phi)
    pts[:, 1] = y0 + a * np.sin(alpha) * np.cos(phi) + b * np.cos(alpha) * np.sin(phi)
    return pts


def b_field_circle(I, x, y, x0, y0):
    """
    magnetic field of infinite straight wire.
    """
    b_vec = np.zeros((len(x), 3))
    b_vec[:, 0] = -0.0000002 * I * (y - y0) / ((y - y0) ** 2 + (x - x0) ** 2)
    b_vec[:, 1] = 0.0000002 * I * (x - x0) / ((y - y0) ** 2 + (x - x0) ** 2)
    b_vec[:, 2] = 0

    return b_vec


def b_field_ellipse(I, x, y, x0, y0, lamb, alpha):
    """
    Magnetic field components in the elliptic field case:
    (x, y)... position of the observer
    I... total current through this point
    a_t = a/b... where a, b  are the major and minor semi axes of the ellipse
    (x0, y0)... position of the source in the x-y plane
    alpha... the tilt of the ellipse (angle between x axis and major axis)
    """
    mu0 = 4 * np.pi * 10 ** (-7)
    alpha = alpha % (2 * np.pi)

    a_t = (1 + lamb) / (1 - lamb)

    b = np.sqrt(((y - y0) * np.cos(alpha) - (x - x0) * np.sin(alpha)) ** 2
                + ((y - y0) * np.sin(alpha) + (x - x0) * np.cos(alpha)) ** 2 / a_t ** 2)
    a = a_t * b

    phi = np.arctan2(a * (np.cos(alpha) * (y - y0) - np.sin(alpha) * (x - x0)),
                     b * (np.cos(alpha) * (x - x0) + np.sin(alpha) * (y - y0))) % (2 * np.pi)

    lamb = (a_t - 1) / (a_t + 1)
    u_ell = (a + b) * np.pi * (1 + lamb ** 2 / 4)  # + O(lamb**4)

    b_mag = I * mu0 / u_ell

    dx_dphi = (-a_t * np.cos(alpha) * np.sin(phi) - np.sin(alpha) * np.cos(phi))
    dy_dphi = (-a_t * np.sin(alpha) * np.sin(phi) + np.cos(alpha) * np.cos(phi))
    abs_dr_dphi = np.sqrt(np.cos(phi) ** 2 + a_t ** 2 * np.sin(phi) ** 2)

    e_phi_x = dx_dphi / abs_dr_dphi
    e_phi_y = dy_dphi / abs_dr_dphi

    b_vec = np.zeros((len(x), 3))

    b_vec[:, 0] = e_phi_x * b_mag
    b_vec[:, 1] = e_phi_y * b_mag
    b_vec[:, 2] = 0

    return b_vec


def model_function_geometry_circle(x_data, I, x0, y0):
    x = x_data[:,0]
    y = x_data[:,1]
    z = x_data[:,2]


    b_vec_comp = b_field_circle(I, x, y, x0, y0)[:,0]

    return b_vec_comp

def model_function_geometry_ellipse(x_data, I, x0, y0, lamb, alpha):
    x = x_data[:,0]
    y = x_data[:,1]
    z = x_data[:,2]


    lamb1 = np.arctan(lamb) / np.pi * 2

    b_vec_comp = b_field_ellipse(I, x, y, x0, y0, lamb1, alpha)[:,0]

    return b_vec_comp