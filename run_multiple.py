import numpy as np
from matplotlib import pyplot as plt
from numpy import random
import math
import util
from datetime import datetime
import os
import hyper_geo
import heights

import scipy
from matplotlib import cm

today = datetime.now()


name = str(today.strftime('%Y%m%d_%H%M%S'))

save_dir = './plots/run_' + name

if os.path.isdir('./plots/run_' + name) == False:
    os.mkdir('./plots/run_' + name)

#This part randomly generates the 3D scene
f = plt.figure(figsize=(10,10))

no_points = 20
#random.seed(43)

#f = random.uniform(2000, 5000) #focal length
t1 = 940.0 #focal center
t2 = 560.0

h = np.random.normal(1.6, 0.1, no_points - 1)
n = [random.uniform(-5, 5), random.uniform(1, 11), random.uniform(-5, 5)] #normal vecetor
n = n / np.linalg.norm(n)
p = [random.uniform(-5, 5), random.uniform(-5, 5) - 10, random.uniform(-5, 5)] #plane center

#print(f, " gt focal")
#print(n,  " gt normal ")

phs, pht, phu = [], [], []

pixel_error = np.random.normal(0, 3, (2, no_points - 1))
for j in range(0, 10):

    fig, ax0 = plt.subplots(1, 1)

    fig1, ax1 = plt.subplots(1, 1)

    fig2, ax2 = plt.subplots(1, 1)

    fig3, ax3 = plt.subplots(1, 1)

    f = random.uniform(2000, 5000) #focal length
    c = np.array([[f, 0, t1], [0, f, t2], [0, 0, 1]]) # intrinsic camera matrix
    # This part generates the random ankle coordinates of people
    ax, ay, az = [], [], []
    hx, hy, hz = [], [], []
    au, av, aw = [], [], []
    hu, hv, hw = [], [], []

    for i in range(0, no_points-1):
        ax.append(random.uniform(2, 7))
        az.append(random.uniform(2, 7))
        ay.append(p[1] + (-n[0]*(ax[i]-p[0])-n[2]*(az[i]-p[2]))/n[1])


    for i in range(0, no_points-1):
        hx.append(([ax[i], ay[i], az[i]]+h[i]*n)[0])
        hy.append(([ax[i], ay[i], az[i]]+h[i]*n)[1])
        hz.append(([ax[i], ay[i], az[i]]+h[i]*n)[2])

    #Applys perspective matrix to transform 3D into 2D points
    for i in range(0, no_points-1):
        au.append(np.dot(c, np.array([[ax[i]], [ay[i]], [az[i]]]))[0].item())
        av.append(np.dot(c, np.array([[ax[i]], [ay[i]], [az[i]]]))[1].item())
        aw.append(np.dot(c, np.array([[ax[i]], [ay[i]], [az[i]]]))[2].item())
        au[i] = au[i]/aw[i]
        av[i] = av[i]/aw[i]
        aw[i] = 1

    for i in range(0, no_points-1):
        hu.append(np.dot(c, np.array([[hx[i]], [hy[i]], [hz[i]]]))[0].item())
        hv.append(np.dot(c, np.array([[hx[i]], [hy[i]], [hz[i]]]))[1].item())
        hw.append(np.dot(c, np.array([[hx[i]], [hy[i]], [hz[i]]]))[2].item())

        hu[i] = hu[i]/hw[i]
        hv[i] = hv[i]/hw[i]
        hw[i] = 1

    #Constructs DLT matrix
    points = np.array([ax, ay, az])

    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))

    left = svd[0]

    normal = left[:, -1]

    normal = normal/np.linalg.norm(normal)

    termination = 10000

    vargs = [au, hu, av, hv, t1, t2]
    x0 = 1.6*np.ones(no_points - 1)

    #best_init = 1.6*np.ones(no_points - 1)
    '''
    best_error = np.inf

    for b in range(1000):
        test_rand = np.random.normal(1.6, 1, no_points - 1)
        result, err, fx_array, fy_array = heights.pytorch_average_plane_optimize_height(test_rand, vargs, term = 1)

        #print(err[0], " hiiasdadsaasd")
        if best_error > err[0]:
            best_error = err[0]
            x0 = test_rand
    '''
    params = []
    error_array = []

    X = []
    Y = []
    Z = []
    V = []

    x0 = np.random.normal(1.6, 0, no_points - 1)

    #result, err, fx_array, fy_array = heights.pytorch_average_plane_optimize_height(x0, vargs, term = termination)
    result, err, fx_array, fy_array, best_cal, best_error = heights.pytorch_average_random_optimize_height(x0, vargs, term = termination)
    print("******************************88")
    print(f, " focal length")
    print(n, " normal")
    print(p, " point")
    print(best_cal, ' best cal')
    print(best_error, ' best error')
    err_array = []

    fx_error_array = []
    fy_error_array = []

    for i in range(len(err)):
        err_array.append(err[i])

    for i in range(len(fx_array)):
        fx_error_array.append(np.abs(fx_array[i] - f))

    for i in range(len(fy_array)):
        fy_error_array.append(np.abs(fy_array[i] - f))

    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.plot(err_array)
    fig1.savefig('./plots/run_' + name + "/" + 'error_curve_' + str(j) + '_.png')

    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.plot(fx_error_array)
    fig2.savefig('./plots/run_' + name + "/" + 'fx_curve_' + str(j) + '_.png')

    ax3.set_yscale('log')
    ax3.set_xscale('log')
    ax3.plot(fy_error_array)
    fig3.savefig('./plots/run_' + name + "/" + 'fy_curve_' + str(j) + '_.png')

    plt.close('all')