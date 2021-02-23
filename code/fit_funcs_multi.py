import numpy as np


def f_b_field_prod_fit(x, I, a_t, x0, y0, alpha, *args):
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

    b_field_vec = np.zeros((3, len(x)))
    b_field_vec[0, :] = e_phi_x / norm * b_mag
    b_field_vec[1, :] = e_phi_y / norm * b_mag
    b_field_vec[2, :] = 0

    b_field_det = args[0]

    b_mag_det = np.sqrt(b_field_det[0, :] ** 2 + b_field_det[1, :] ** 2)

    angle = b_field_vec[0, :] * b_field_det[0, :] + b_field_vec[1, :] * b_field_det[1, :] - b_mag * b_mag_det

    return angle
