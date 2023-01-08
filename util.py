from scipy import special
import random as rand
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import mpmath
import torch

import scipy
import hyper_geo

'''
p is the point to project
p0 is a point on the plane (one of the points defining the quadrilateral, their centroid...)
u is the unit normal vector of the plane
'''
def plane_proj(u, p, p0):
    #projected_point = p - torch.dot(p - p0, u) * u
    projected_point = p - torch.mul((torch.unsqueeze(u, dim = 0) @  (p - torch.unsqueeze(p0, dim = 1).repeat(1, p.shape[1]))), torch.transpose(torch.unsqueeze(u, dim = 0).repeat(p.shape[1], 1), 0 , 1))

    return projected_point

def ankle_calibration(p_2d, p_3d, t1, t2):

    fx_array = []
    fy_array = []

    for i in range(p_3d.shape[1]):

        fx_array.append(p_3d[2][i]*(p_2d[i][0] - t1)/p_3d[0][i])
        fy_array.append(p_3d[2][i]*(p_2d[i][1] - t2)/p_3d[1][i])

    #print(fx_array)
    #print(fy_array)
    fx = torch.stack(fx_array).mean()
    fy = torch.stack(fy_array).mean()

    #print(fx, fy, " focals")
    return fx, fy

def plane_ray_intersection_torch(x_imcoord, y_imcoord, cam_inv, normal, init_point):
    """ 
    Recovers the 3d coordinates from 2d by computing the intersection between the 2d point's ray and the plane
    
    Parameters: x_imcoord: float list or np.array
                    x coordinates in camera coordinates
                y_imcoord: float list or np.array
                    y coordinates in camera coordinates
                cam_inv: (3,3) np.array
                    inverse camera intrinsic matrix
                normal: (3,) np.array
                    normal vector of the plane
                init_point: (3,) np.array
                    3d point on the plane used to initalize the ground plane
    Returns:    ray: (3,) np.array
                    3d coordinates of (x_imcoord, y_imcoord)
    """
    point_2d = torch.stack((x_imcoord, y_imcoord, torch.ones(x_imcoord.shape[0])))
    
    ray = cam_inv @ point_2d
    #print(torch.unsqueeze(normal, dim = 0).shape, ray.shape, "heqwqewqeweqweqw qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq")
    normal_dot_ray = torch.matmul(torch.unsqueeze(normal, dim = 0), ray) + 1e-15
    
    scale = abs(torch.div(torch.dot(normal, init_point).repeat(x_imcoord.shape[0]), normal_dot_ray))
    return scale*ray

#This function is used to plot the closed form pdf
def estimator_closed_form_pdf(u, v, t_1, t_2, m1, v1, start, end, itr):

    w_array = list(np.linspace(start, end, itr))

    f_squared_array = []

    for i in range(len(w_array)):
        f_squared = closed_form_ratio(u, v, t_1, t_2, m1, v1, w_array[i])
        
        f_squared_array.append(f_squared)

    return w_array, f_squared_array

#computes N1/N2, which are both normal
def closed_form_ratio(u, v, t_1, t_2, u_x, s_x, w):
    
    a, b, c ,d = coef(u, v, t_1, t_2)

    p, m1, m2, var1, var2 = dependent_normal_correlation(u_x, s_x, a, b, c ,d)
    #print(p, m1, m2, var1, var2, " HELOOOOSADASD")

    f_squared = dependent_normal_pdf(w, m1, m2, np.sqrt(var1), np.sqrt(var2), p)
    return float(f_squared)

# computes coefficients based on keypoints
def coef(u, v, t_1, t_2):

    v_11 = v[0][0]
    v_12 = v[0][1]
    v_13 = v[0][2]

    v_21 = v[1][0]
    v_22 = v[1][1]
    v_23 = v[1][2]

    #########################

    u_11 = u[0][0]
    u_12 = u[0][1]
    u_13 = u[0][2]

    u_21 = u[1][0]
    u_22 = u[1][1]
    u_23 = u[1][2]

    h1 = 1
    h2 = 1
    h3 = 1

    c1 = (-t_1*u_11*v_12*v_13 + t_1*u_11*v_12*v_23 + t_1*u_11*v_13*v_22 - t_1*u_11*v_22*v_23 + t_1*u_12*v_11*v_13 - t_1*u_12*v_11*v_23 - t_1*u_12*v_13*v_21 + t_1*u_12*v_21*v_23 + t_1*u_21*v_12*v_13 - t_1*u_21*v_12*v_23 - t_1*u_21*v_13*v_22 + t_1*u_21*v_22*v_23 - t_1*u_22*v_11*v_13 + t_1*u_22*v_11*v_23 + t_1*u_22*v_13*v_21 - t_1*u_22*v_21*v_23 + u_11*u_12*v_13*v_21 - u_11*u_12*v_13*v_22 - u_11*u_12*v_21*v_23 + u_11*u_12*v_22*v_23 + u_11*u_22*v_12*v_13 - u_11*u_22*v_12*v_23 - u_11*u_22*v_13*v_21 + u_11*u_22*v_21*v_23 - u_12*u_21*v_11*v_13 + u_12*u_21*v_11*v_23 + u_12*u_21*v_13*v_22 - u_12*u_21*v_22*v_23 + u_21*u_22*v_11*v_13 - u_21*u_22*v_11*v_23 - u_21*u_22*v_12*v_13 + u_21*u_22*v_12*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c2 = (-t_2*u_11*v_12*v_13 + t_2*u_11*v_12*v_23 + t_2*u_11*v_13*v_22 - t_2*u_11*v_22*v_23 + t_2*u_12*v_11*v_13 - t_2*u_12*v_11*v_23 - t_2*u_12*v_13*v_21 + t_2*u_12*v_21*v_23 + t_2*u_21*v_12*v_13 - t_2*u_21*v_12*v_23 - t_2*u_21*v_13*v_22 + t_2*u_21*v_22*v_23 - t_2*u_22*v_11*v_13 + t_2*u_22*v_11*v_23 + t_2*u_22*v_13*v_21 - t_2*u_22*v_21*v_23 + u_11*v_12*v_13*v_21 - u_11*v_12*v_21*v_23 - u_11*v_13*v_21*v_22 + u_11*v_21*v_22*v_23 - u_12*v_11*v_13*v_22 + u_12*v_11*v_22*v_23 + u_12*v_13*v_21*v_22 - u_12*v_21*v_22*v_23 - u_21*v_11*v_12*v_13 + u_21*v_11*v_12*v_23 + u_21*v_11*v_13*v_22 - u_21*v_11*v_22*v_23 + u_22*v_11*v_12*v_13 - u_22*v_11*v_12*v_23 - u_22*v_12*v_13*v_21 + u_22*v_12*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c3 = (u_11*v_12*v_13 - u_11*v_12*v_23 - u_11*v_13*v_22 + u_11*v_22*v_23 - u_12*v_11*v_13 + u_12*v_11*v_23 + u_12*v_13*v_21 - u_12*v_21*v_23 - u_21*v_12*v_13 + u_21*v_12*v_23 + u_21*v_13*v_22 - u_21*v_22*v_23 + u_22*v_11*v_13 - u_22*v_11*v_23 - u_22*v_13*v_21 + u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    
    #ratio
    c4 = (-h1*u_12*v_13*v_21 + h1*u_12*v_13*v_22 + h1*u_12*v_21*v_23 - h1*u_12*v_22*v_23 + h1*u_21*v_12*v_13 - h1*u_21*v_12*v_23 - h1*u_21*v_13*v_22 + h1*u_21*v_22*v_23 - h1*u_22*v_12*v_13 + h1*u_22*v_12*v_23 + h1*u_22*v_13*v_21 - h1*u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c5 = (-h2*u_11*v_13*v_21 + h2*u_11*v_13*v_22 + h2*u_11*v_21*v_23 - h2*u_11*v_22*v_23 + h2*u_21*v_11*v_13 - h2*u_21*v_11*v_23 - h2*u_21*v_13*v_22 + h2*u_21*v_22*v_23 - h2*u_22*v_11*v_13 + h2*u_22*v_11*v_23 + h2*u_22*v_13*v_21 - h2*u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)

    a = (-c1*c4*(u_11 - t_1) - c2*c4*(v_11 - t_2))
    b = (c1*c5*(u_12 - t_1) + c2*c5*(v_12 - t_2))
    c = c3*c4
    d = c3*c5

    return a, b, c, d

#computes the correlation coefficient of the two normals
def dependent_normal_correlation(m1, s1, a, b, c, d):
    #figure out E(X^2)
    var1 = a**2*s1**2 + b**2*s1**2
    var2 = c**2*s1**2 + d**2*s1**2

    cov_xx = s1**2 + s1**2

    covariance = -1*(a*c*s1**2 + b*(-d)*s1**2 + (b*c + a*(-d))*cov_xx)
    mean1 = a*m1 + b*m1
    mean2 = c*m1 - d*m1

    return covariance/(var1*var2), mean1, mean2, var1, var2

#implements PDF descriped in Pham-Gia et al.
def dependent_normal_pdf(w, u_x, u_y, s_x, s_y, p):

    coef = (2*(1 - p**2)*s_x**2*s_y**2)/(s_y**2*w**2 - 2*p*s_x*s_y*w + s_x**2)

    hyp_geo = mpmath.hyp1f1(1, 1/2, theta2(w, u_x, u_y, s_x, s_y, p))

    coef1 = k2(u_x, u_y, s_x, s_y, p)
    result = coef1*coef*hyp_geo

    return result

def theta2(w, u_x, u_y, s_x, s_y, p):

    num = (-s_y**2*u_x*w + p*s_x*s_y*(u_y*w + u_x) - u_y*s_x**2)**2
    den = (2*s_x**2*s_y**2)*(1 - p**2)*(s_y**2*w**2 - 2*p*s_x*s_y*w + s_x**2)
    return num/den

def k2(u_x, u_y, s_x, s_y, p):
    
    coef = 1/(2*np.pi*s_x*s_y*(1 - p**2)**0.5)

    num = s_y**2*u_x**2 - 2*p*s_x*s_y*u_x*u_y + u_y**2*s_x**2 
    den = 2*(1 - p**2)*s_x**2*s_y**2
    return coef*np.exp(-num/den)

def normal_pdf(bins, mu, sigma):

    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) )

# This function implements method 1 in the report, the histogram method
def monte_carlo_estimator(u, v, t_1, t_2, m1, v1, m2, v2, itr):

    height_random = np.random.normal(m1, v1, size = (itr, 3)) # get N by 3 samples
    detection_error = np.random.normal(m2, v2, size = (itr, 12)) # GET N by 12 samples

    f_array = []
    n_array = []
    z1_array = []
    z2_array = []
    z3_array = []

    c1_array = []
    c2_array = []
    c3_array = []
    c4_array = []
    c5_array = []
    f_squared_array = []

    for i in range(itr):
        c1, c2, c3, c4, c5, f, f_squared, n, z1, z2, z3 = closed_form_error(u, v, t_1, t_2, height_random[i], detection_error[i]) # compute camera parameters
        
        f_squared_array.append(f_squared)
        f_array.append(f)
        n_array.append(n)
        z1_array.append(z1)
        z2_array.append(z2)
        z3_array.append(z3)

        c1_array.append(c1)
        c2_array.append(c2)
        c3_array.append(c3)
        c4_array.append(c4)
        c5_array.append(c5)
    #print(f_array, " f _ arrat")
    return f_array, f_squared_array, np.array(n_array), z1_array, z2_array, z3_array, c1_array, c2_array, c3_array, c4_array, c5_array

#implements the random variable for focal length
def closed_form_error(u, v, t_1, t_2, h, detection_error = list(np.zeros(12))):
    
    h1 = h[0]
    h2 = h[1]
    h3 = h[2]
    #print(np.array(v).shape, " shape")
    v_11 = v[0][0] + detection_error[0]
    v_12 = v[0][1] + detection_error[1]
    v_13 = v[0][2] + detection_error[2]

    v_21 = v[1][0] + detection_error[3]
    v_22 = v[1][1] + detection_error[4]
    v_23 = v[1][2] + detection_error[5]

    #########################

    u_11 = u[0][0] + detection_error[6]
    u_12 = u[0][1] + detection_error[7]
    u_13 = u[0][2] + detection_error[8]

    u_21 = u[1][0] + detection_error[9]
    u_22 = u[1][1] + detection_error[10]
    u_23 = u[1][2] + detection_error[11]

    c1 = (-t_1*u_11*v_12*v_13 + t_1*u_11*v_12*v_23 + t_1*u_11*v_13*v_22 - t_1*u_11*v_22*v_23 + t_1*u_12*v_11*v_13 - t_1*u_12*v_11*v_23 - t_1*u_12*v_13*v_21 + t_1*u_12*v_21*v_23 + t_1*u_21*v_12*v_13 - t_1*u_21*v_12*v_23 - t_1*u_21*v_13*v_22 + t_1*u_21*v_22*v_23 - t_1*u_22*v_11*v_13 + t_1*u_22*v_11*v_23 + t_1*u_22*v_13*v_21 - t_1*u_22*v_21*v_23 + u_11*u_12*v_13*v_21 - u_11*u_12*v_13*v_22 - u_11*u_12*v_21*v_23 + u_11*u_12*v_22*v_23 + u_11*u_22*v_12*v_13 - u_11*u_22*v_12*v_23 - u_11*u_22*v_13*v_21 + u_11*u_22*v_21*v_23 - u_12*u_21*v_11*v_13 + u_12*u_21*v_11*v_23 + u_12*u_21*v_13*v_22 - u_12*u_21*v_22*v_23 + u_21*u_22*v_11*v_13 - u_21*u_22*v_11*v_23 - u_21*u_22*v_12*v_13 + u_21*u_22*v_12*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c2 = (-t_2*u_11*v_12*v_13 + t_2*u_11*v_12*v_23 + t_2*u_11*v_13*v_22 - t_2*u_11*v_22*v_23 + t_2*u_12*v_11*v_13 - t_2*u_12*v_11*v_23 - t_2*u_12*v_13*v_21 + t_2*u_12*v_21*v_23 + t_2*u_21*v_12*v_13 - t_2*u_21*v_12*v_23 - t_2*u_21*v_13*v_22 + t_2*u_21*v_22*v_23 - t_2*u_22*v_11*v_13 + t_2*u_22*v_11*v_23 + t_2*u_22*v_13*v_21 - t_2*u_22*v_21*v_23 + u_11*v_12*v_13*v_21 - u_11*v_12*v_21*v_23 - u_11*v_13*v_21*v_22 + u_11*v_21*v_22*v_23 - u_12*v_11*v_13*v_22 + u_12*v_11*v_22*v_23 + u_12*v_13*v_21*v_22 - u_12*v_21*v_22*v_23 - u_21*v_11*v_12*v_13 + u_21*v_11*v_12*v_23 + u_21*v_11*v_13*v_22 - u_21*v_11*v_22*v_23 + u_22*v_11*v_12*v_13 - u_22*v_11*v_12*v_23 - u_22*v_12*v_13*v_21 + u_22*v_12*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c3 = (u_11*v_12*v_13 - u_11*v_12*v_23 - u_11*v_13*v_22 + u_11*v_22*v_23 - u_12*v_11*v_13 + u_12*v_11*v_23 + u_12*v_13*v_21 - u_12*v_21*v_23 - u_21*v_12*v_13 + u_21*v_12*v_23 + u_21*v_13*v_22 - u_21*v_22*v_23 + u_22*v_11*v_13 - u_22*v_11*v_23 - u_22*v_13*v_21 + u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c4 = (-h1*u_12*v_13*v_21 + h1*u_12*v_13*v_22 + h1*u_12*v_21*v_23 - h1*u_12*v_22*v_23 + h1*u_21*v_12*v_13 - h1*u_21*v_12*v_23 - h1*u_21*v_13*v_22 + h1*u_21*v_22*v_23 - h1*u_22*v_12*v_13 + h1*u_22*v_12*v_23 + h1*u_22*v_13*v_21 - h1*u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    c5 = (-h2*u_11*v_13*v_21 + h2*u_11*v_13*v_22 + h2*u_11*v_21*v_23 - h2*u_11*v_22*v_23 + h2*u_21*v_11*v_13 - h2*u_21*v_11*v_23 - h2*u_21*v_13*v_22 + h2*u_21*v_22*v_23 - h2*u_22*v_11*v_13 + h2*u_22*v_11*v_23 + h2*u_22*v_13*v_21 - h2*u_22*v_21*v_23)/(h3*u_11*v_12*v_21 - h3*u_11*v_12*v_23 - h3*u_11*v_21*v_22 + h3*u_11*v_22*v_23 - h3*u_12*v_11*v_22 + h3*u_12*v_11*v_23 + h3*u_12*v_21*v_22 - h3*u_12*v_21*v_23 - h3*u_21*v_11*v_12 + h3*u_21*v_11*v_22 + h3*u_21*v_12*v_23 - h3*u_21*v_22*v_23 + h3*u_22*v_11*v_12 - h3*u_22*v_11*v_23 - h3*u_22*v_12*v_21 + h3*u_22*v_21*v_23)
    
    f_squared = ((-c1*(c4*(u_11 - t_1) - c5*(u_12 - t_1)) - c2*(c4*(v_11 - t_2) - c5*(v_12 - t_2)))/(c3*(c4 - c5)))
    f = np.sqrt(np.absolute(f_squared))

    n1 = c1
    n2 = c2
    n3 = f*c3

    n = np.array([n1, n2, n3])
    lda = np.linalg.norm(n)
    n = n/lda

    z1 = f*c4/lda
    z2 = f*c5/lda
    z3 = -1*f/lda

    return c1, c2, c3, c4, c5, f, f_squared, n, z1, z2, z3
    
def random_combination(image_index, num_points, termination_cond):
    '''
    gets len(image_index) choose num_points combinations of detections, until the number of combinations exceed termination_comd

    Parameters: image_index: list
                    indices of detections in datastore
                num_points: int
                    number of points to solve DLT equations
                termination_cond: int
                    maximum number of combinations
    Returns: samples: list
                Combinations of detections
    '''
    total_comb = int(special.comb(len(image_index),num_points))
    samples = set()
    while len(samples) < termination_cond and len(samples) < total_comb:
        samples.add(tuple(sorted(rand.sample(image_index, num_points))))

    return list(samples)