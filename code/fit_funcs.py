import numpy as np
from scipy.special import hyp2f1


def f_b_field(x, I, y):
    """
    x-component of the magnetic field of infinite straight wire.
    """
    return 0.0000002 * I * y / (y * y + x * x)


def f_b_field_off(x, I, y, x_s):
    """
    x-component of the magnetic field of infinite straight wire shifted by x_s.
    """

    return 0.0000002 * I * y / (y * y + (x - x_s) ** 2)  # *u_circle


def f_b_field_elliptic_pts(x, y, I, a_t, x0, y0, alpha):
    """
    Magnetic field components in the elliptic field case:
    (x, y)... position of the observer
    I... total current through this point
    a_t = a/b... where a, b  are the major and minor semi axes of the ellipse
    (x0, y0)... position of the source in the x-y plane
    alpha... the tilt of the ellipse (angle between x axis and major axis)
    """
    mu0 = 4 * np.pi * 10 ** (-7)

    b_vec = np.sqrt(((y - y0) * np.cos(alpha) - (x - x0) * np.sin(alpha)) ** 2
                    + ((y - y0) * np.sin(alpha) + (x - x0) * np.cos(alpha)) ** 2 / a_t ** 2)
    a_vec = a_t * b_vec

    phi = np.arctan2(a_vec * (np.cos(alpha) * (y - y0) - np.sin(alpha) * (x - x0)),
                     b_vec * (np.cos(alpha) * (x - x0) + np.sin(alpha) * (y - y0))) % (2 * np.pi)

    lamb = (a_t - 1) / (a_t + 1)
    u_ell = np.pi * b_vec * (1 + a_t) * (1 + 3 * lamb ** 2 / (10 + np.sqrt(4 - 3 * lamb ** 2)))

    b_mag = I * mu0 / u_ell

    e_phi_x = (-a_t * np.cos(alpha) * np.sin(phi) - np.sin(alpha) * np.cos(phi))
    e_phi_y = (-a_t * np.sin(alpha) * np.sin(phi) + np.cos(alpha) * np.cos(phi))
    norm = np.sqrt(np.cos(phi) ** 2 + a_t ** 2 * np.sin(phi) ** 2)

    b_vec = np.zeros((3, len(x)))
    b_vec[0, :] = e_phi_x / norm * b_mag
    b_vec[1, :] = e_phi_y / norm * b_mag
    b_vec[2, :] = 0

    return b_vec


def f_b_field_elliptic_field_pts(x, y, I, a_t, x0, y0, alpha):
    """
    Magnetic field components in the elliptic field case:
    (x, y)... position of the observer
    I... total current through this point
    a_t = a/b... where a, b  are the major and minor semi axes of the ellipse
    (x0, y0)... position of the source in the x-y plane
    alpha... the tilt of the ellipse (angle between x axis and major axis)
    """
    mu0 = 4 * np.pi * 10 ** (-7)

    b_vec = np.sqrt(((y - y0) * np.cos(alpha) - (x - x0) * np.sin(alpha)) ** 2
                    + ((y - y0) * np.sin(alpha) + (x - x0) * np.cos(alpha)) ** 2 / a_t ** 2)
    a_vec = a_t * b_vec

    phi = np.arctan2(a_vec * (np.cos(alpha) * (y - y0) - np.sin(alpha) * (x - x0)),
                     b_vec * (np.cos(alpha) * (x - x0) + np.sin(alpha) * (y - y0))) % (2 * np.pi)

    lamb = (a_t - 1) / (a_t + 1)
    u_ell = np.pi * b_vec * (1 + a_t) * (1 + 3 * lamb ** 2 / (10 + np.sqrt(4 - 3 * lamb ** 2)))

    b_mag = I * mu0 / u_ell

    e_phi_x = (-a_t * np.cos(alpha) * np.sin(phi) - np.sin(alpha) * np.cos(phi))
    e_phi_y = (-a_t * np.sin(alpha) * np.sin(phi) + np.cos(alpha) * np.cos(phi))
    norm = np.sqrt(np.cos(phi) ** 2 + a_t ** 2 * np.sin(phi) ** 2)

    b_vec = np.zeros((3, len(x)))
    b_vec[0, :] = e_phi_x / norm
    b_vec[1, :] = e_phi_y / norm
    b_vec[2, :] = 0

    return b_vec * b_mag


def u_exact(a, b):
    t = ((a - b) / (a + b)) ** 2
    return np.pi * (a + b) * hyp2f1(-0.5, -0.5, 1, t)


def f_b_field_elliptic_fit_x(x, I, a_t, x0, y0, alpha):
    """
    Magnetic field components in the elliptic field case:
    (x, y)... position of the observer
    I... total current through this point
    a_t = a/b... where a, b  are the major and minor semi axes of the ellipse
    (x0, y0)... position of the source in the x-y plane
    alpha... the tilt of the ellipse (angle between x axis and major axis)
    """
    y = 0
    mu0 = 4 * np.pi * 10 ** (-7)

    b_vec = np.sqrt(((y - y0) * np.cos(alpha) - (x - x0) * np.sin(alpha)) ** 2
                    + ((y - y0) * np.sin(alpha) + (x - x0) * np.cos(alpha)) ** 2 / a_t ** 2)
    a_vec = a_t * b_vec

    phi = np.arctan2(a_vec * (np.cos(alpha) * (y - y0) - np.sin(alpha) * (x - x0)),
                     b_vec * (np.cos(alpha) * (x - x0) + np.sin(alpha) * (y - y0))) % (2 * np.pi)

    lamb = (a_t - 1) / (a_t + 1)
    u_ell = np.pi * b_vec * (1 + a_t) * (1 + 3 * lamb ** 2 / (10 + np.sqrt(4 - 3 * lamb ** 2)))

    b_mag = I * mu0 / u_ell

    e_phi_x = (-a_t * np.cos(alpha) * np.sin(phi) - np.sin(alpha) * np.cos(phi))
    e_phi_y = (-a_t * np.sin(alpha) * np.sin(phi) + np.cos(alpha) * np.cos(phi))
    norm = np.sqrt(np.cos(phi) ** 2 + a_t ** 2 * np.sin(phi) ** 2)

    b_vec = np.zeros((3, len(x)))
    b_vec[0] = e_phi_x / norm * b_mag
    b_vec[1] = e_phi_y / norm * b_mag
    b_vec[2, :] = 0

    return b_vec[0]

def f_b_field_elliptic_fit(x, I, a_t, x0, y0, alpha):
    """
    Magnetic field components in the elliptic field case:
    (x, y)... position of the observer
    I... total current through this point
    a_t = a/b... where a, b  are the major and minor semi axes of the ellipse
    (x0, y0)... position of the source in the x-y plane
    alpha... the tilt of the ellipse (angle between x axis and major axis)
    """
    y = 0
    mu0 = 4 * np.pi * 10 ** (-7)

    b_vec = np.sqrt(((y - y0) * np.cos(alpha) - (x - x0) * np.sin(alpha)) ** 2
                    + ((y - y0) * np.sin(alpha) + (x - x0) * np.cos(alpha)) ** 2 / a_t ** 2)
    a_vec = a_t * b_vec

    phi = np.arctan2(a_vec * (np.cos(alpha) * (y - y0) - np.sin(alpha) * (x - x0)),
                     b_vec * (np.cos(alpha) * (x - x0) + np.sin(alpha) * (y - y0))) % (2 * np.pi)

    lamb = (a_t - 1) / (a_t + 1)
    u_ell = np.pi * b_vec * (1 + a_t) * (1 + 3 * lamb ** 2 / (10 + np.sqrt(4 - 3 * lamb ** 2)))

    b_mag = I * mu0 / u_ell

    e_phi_x = (-a_t * np.cos(alpha) * np.sin(phi) - np.sin(alpha) * np.cos(phi))
    e_phi_y = (-a_t * np.sin(alpha) * np.sin(phi) + np.cos(alpha) * np.cos(phi))
    norm = np.sqrt(np.cos(phi) ** 2 + a_t ** 2 * np.sin(phi) ** 2)

    b_vec = np.zeros((3, len(x)))
    b_vec[0] = e_phi_x / norm * b_mag
    b_vec[1] = e_phi_y / norm * b_mag
    b_vec[2, :] = 0

    return b_vec

def ellipse_pts(phi, x0, y0, alpha, a, b):
    pts = np.zeros((2, len(phi)))
    pts[0, :] = x0 + a * np.cos(alpha) * np.cos(phi) - b * np.sin(alpha) * np.sin(phi)
    pts[1, :] = y0 + a * np.sin(alpha) * np.cos(phi) + b * np.cos(alpha) * np.sin(phi)
    return pts

def ellipse_tang_vecs(phi, alpha, a, b):
    tang_x = - a * np.cos(alpha) * np.sin(phi) - b * np.sin(alpha) * np.cos(phi)
    tang_y = - a * np.sin(alpha) * np.sin(phi) + b * np.cos(alpha) * np.cos(phi)
    norm = np.sqrt(tang_x ** 2 + tang_y ** 2)
    return np.array([tang_x / norm, tang_y / norm])
