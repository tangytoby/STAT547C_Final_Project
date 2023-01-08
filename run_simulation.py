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

today = datetime.now()


name = str(today.strftime('%Y%m%d_%H%M%S'))

save_dir = './plots/run_' + name

if os.path.isdir('./plots/run_' + name) == False:
    os.mkdir('./plots/run_' + name)

#This part randomly generates the 3D scene
f = plt.figure(figsize=(10,10))

no_points = 4
random.seed(43)

f = random.uniform(2000, 5000) #focal length
t1 = 940.0 #focal center
t2 = 560.0
h = 1.6 #height

n = [random.uniform(-5, 5), random.uniform(1, 11), random.uniform(-5, 5)] #normal vecetor
n = n / np.linalg.norm(n)
p = [random.uniform(-5, 5), random.uniform(-5, 5) - 10, random.uniform(-5, 5)] #plane center
q = (p+h*n)[0]
c = np.array([[f, 0, t1], [0, f, t2], [0, 0, 1]]) # intrinsic camera matrix

phs, pht, phu = [], [], []

fig, ax0 = plt.subplots(1, 1)

fig1, ax1 = plt.subplots(1, 1)

fig2, ax2 = plt.subplots(1, 1)

for j in range(0, 1):

    # This part generates the random ankle coordinates of people
    ax, ay, az = [], [], []
    hx, hy, hz = [], [], []
    au, av, aw = [], [], []
    hu, hv, hw = [], [], []
    sub = []

    for i in range(0, no_points-1):
        ax.append(random.uniform(2, 7))
        az.append(random.uniform(2, 7))
        ay.append(p[1] + (-n[0]*(ax[i]-p[0])-n[2]*(az[i]-p[2]))/n[1])


    for i in range(0, no_points-1):
        hx.append(([ax[i], ay[i], az[i]]+h*n)[0])
        hy.append(([ax[i], ay[i], az[i]]+h*n)[1])
        hz.append(([ax[i], ay[i], az[i]]+h*n)[2])
        sub.append([hx[i]-ax[i], hy[i]-ay[i], hz[i]-az[i]])

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

        print(hw[i], " HWWWWWWWWWWWWWWWWWWWWWWWWWWW")
        hu[i] = hu[i]/hw[i]
        hv[i] = hv[i]/hw[i]
        hw[i] = 1

    #Constructs DLT matrix
    print(ax)
    print(ay)
    print(az, " 3D ASDASSDADDDD")

    print(hx)
    print(hy)
    print(hz, " 3D HEADDD ASDASSDADDDD")

    print(au)
    print(av, " 2D ASDASSDADDDD")

    print(hu)
    print(hv, " head 2D ASDASSDADDDD")
    C = np.zeros([6,6], dtype = float)
    C[0] = [0, -1, hv[0], hv[0] - av[0],0,0]
    C[1] = [1, 0, -hu[0], au[0] - hu[0],0,0]
    C[2] = [0, -1, hv[1], 0, hv[1] - av[1], 0]
    C[3] = [1, 0, -hu[1], 0, au[1] - hu[1], 0]
    C[4] = [0, -1, hv[2], 0, 0, hv[2] - av[2]]
    C[5] = [1, 0, -hu[2], 0, 0, au[2] - hu[2]]

    # mean and standard deviation of height normal random variable
    m1 = 1.6
    v1 = 0.1
    # mean and standard deviation of keypoints, set to 0 to simplify problem.
    m2 = 0
    v2 = 0

    termination = 10000

    integral_mean, integral_var = hyper_geo.expectation([au, hu], [av, hv], t1, t2, m1, v1) #computes mean and variance based on closed form pdf

    print(integral_mean, " Closed form mean")
    print(integral_var, " Closed form variance")

    f_avg, f_squared_array, n_avg, z1_avg, z2_avg, z3_avg, c1_array, c2_array, c3_array, c4_array, c5_array = util.monte_carlo_estimator([au, hu], [av, hv], t1, t2, m1, v1, m2, v2, itr = termination) #randomly samples heights to compute f^2
    
    print(np.mean(f_squared_array), " histogram mean")
    print(np.var(f_squared_array), " histogram variance")


    ax0.hist(f_squared_array, termination, density=True)

    fig.savefig('./plots/run_' + name + "/" + 'histogram.png')

    
    w_array, f_squared_ratio_array = util.estimator_closed_form_pdf([au, hu], [av, hv], t1, t2, m1, v1, 1 , 1.4e7, itr = termination) # plots closed form pdf
    ax1.hist(f_squared_array, termination, density=True)
    ax1.scatter(w_array, f_squared_ratio_array, c = 'r')
    fig1.savefig('./plots/run_' + name + "/" + 'pdf_curve_histogram_overlay.png')


    #####################

    ax2.scatter(w_array, f_squared_ratio_array)
    fig2.savefig('./plots/run_' + name + "/" + 'pdf_curve.png')

    plt.close('all')

    print(f, n, p ," CAM PARAMETERS")

    for k in range(0, 3):
        print(ax[k], ay[k], az[k], " ANKLE COORDINATES")