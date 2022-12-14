from scipy.integrate import quad
import numpy as np

import scipy
import util

#This file implements the hypergeometric functions that are needed to compute the PDF in Pham-Gia et al.

#integrand for expectation
def integrand(w, s_x, s_y, u_x, u_y, p):

    coef = (2*(1 - p**2)*s_x**2*s_y**2)/(s_y**2*w**2 - 2*p*s_x*s_y*w + s_x**2)

    hyp_geo = scipy.special.hyp1f1(1, 1/2, util.theta2(w, u_x, u_y, s_x, s_y, p))

    coef1 = util.k2(u_x, u_y, s_x, s_y, p)

    return w*coef1*coef*hyp_geo

#integrand for variance
def variance_integrand(w, s_x, s_y, u_x, u_y, p, ex):
    
    coef = (2*(1 - p**2)*s_x**2*s_y**2)/(s_y**2*w**2 - 2*p*s_x*s_y*w + s_x**2)

    hyp_geo = scipy.special.hyp1f1(1, 1/2, util.theta2(w, u_x, u_y, s_x, s_y, p))

    coef1 = util.k2(u_x, u_y, s_x, s_y, p)

    return (w - ex)**2*coef1*coef*hyp_geo

#computes expectation and variance using quadrature method
def expectation(u, v, t_1, t_2, u_x, s_x):

    a, b, c ,d = util.coef(u, v, t_1, t_2)

    p, m1, m2, var1, var2 = util.dependent_normal_correlation(u_x, s_x, a, b, c ,d)

    I = quad(integrand, -2.5e8, 2.5e8, args=(np.sqrt(var1), np.sqrt(var2), m1, m2, p))

    I1 = quad(variance_integrand, -2.5e8, 2.5e8, args=(np.sqrt(var1), np.sqrt(var2), m1, m2, p, I[0]))

    return I, I1