
def f_b_field(x, I, y):
    """
    y-component of the magnetic field of infinite straight wire.
    """
    return 0.0000002 * I * y / (y * y + x * x)