import numpy as np
import code.fit_funcs_multi as ffm
import code.fit_funcs as ff
from scipy.optimize import curve_fit

def notebook_07():
    y_det = 0
    x_det = np.linspace(-0.5, 0.5, 10)
    x0 = 0.0
    y0 = 2.5
    I = 1.0
    a = 1.0
    b = 1.0
    a_t = a / b
    alpha = 0.0 * np.pi
    b_field_det = ff.f_b_field_elliptic_field_pts(x_det, y_det, I, a_t, x0, y0, alpha)


    p_opt_ell, _ = curve_fit(lambda x_det, I, a_t, x0, y0, alpha:
                             ffm.f_b_field_prod_fit(x_det, I, a_t, x0, y0, alpha, b_field_det), x_det, np.zeros(len(x_det)))

    print(p_opt_ell)



    #print(np.shape(b_pts_det))
