def f_b_field(x, I, y):
    """
    y-component of the magnetic field of infinite straight wire.
    """
    return 0.0000002 * I * y / (y * y + x * x)


def f_b_field_off(x, I, y, x_s):
    """
    y-component of the magnetic field of infinite straight wire shifted by x_s.
    """
    return 0.0000002 * I * y / (y * y + (x - x_s) ** 2)
