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

no_points = 4
random.seed(43)

f = random.uniform(2000, 5000) #focal length
t1 = 940.0 #focal center
t2 = 560.0
h1 = 1.6 #height
h2 = 1.6
h3 = 1.6

h = np.random.normal(1.6, 0, no_points - 1)
n = [random.uniform(-5, 5), random.uniform(1, 11), random.uniform(-5, 5)] #normal vecetor
n = n / np.linalg.norm(n)
p = [random.uniform(-5, 5), random.uniform(-5, 5) - 10, random.uniform(-5, 5)] #plane center

#print(f, " gt focal")
#print(n,  " gt normal ")
c = np.array([[f, 0, t1], [0, f, t2], [0, 0, 1]]) # intrinsic camera matrix

phs, pht, phu = [], [], []

fig, ax0 = plt.subplots(1, 1)

fig1, ax1 = plt.subplots(1, 1)

fig2, ax2 = plt.subplots(1, 1)

pixel_error = np.random.normal(0, 3, (2, no_points - 1))
for j in range(0, 1):

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

        print(hw[i], " HWWWWWWWWWWWWWWWWWWWWWWWWWWW")
        hu[i] = hu[i]/hw[i]
        hv[i] = hv[i]/hw[i]
        hw[i] = 1

    #Constructs DLT matrix
    '''
    print(ax)
    print(ay)
    print(az, " 3D ASDASSDADDDD")
    '''
    points = np.array([ax, ay, az])
    '''
    print(points, " points")
    print(points.shape, " POINTSSSS SHAPE")
    '''
    svd = np.linalg.svd(points - np.mean(points, axis=1, keepdims=True))

    #print(svd, " the svd")
    # Extract the left singular vectors
    left = svd[0]

    normal = left[:, -1]

    normal = normal/np.linalg.norm(normal)
    '''
    print(normal, " normal pred")
    print((points[:, 0] - points[:, 1]).shape, " shae")
    print((np.transpose(points[0]) - np.transpose(points[1])).shape, " sahweae")

    print((points[:, 0] - p).dot(n), " normal gt")
    print((points[:, 0] - p).dot(normal), " normal pred")

    print(hx)
    print(hy)
    print(hz, " 3D HEADDD ASDASSDADDDD")

    print(au)
    print(av, " 2D ASDASSDADDDD")

    print(hu)
    print(hv, " head 2D ASDASSDADDDD")
    '''
    termination = 1

    #MODIFY THIS TO USE MORE THAN 3 POINTS!!!!!!!!!
    # NEW PLAN, TAKE AVERAGE FOCAL AND NORMAL, USE THAT FOR ALL REPROJECTIONS, THEN OPTIMIZE HEIGHT!!!!!!!!!!!
    vargs = [au, hu, av, hv, t1, t2]
    x0 = 1.6*np.ones(no_points - 1)
    #x0 = [h1, h2, h3]

    #c1, c2, c3, c4, c5, f, f_squared, n, z1, z2, z3 = heights.compute_focal([au, hu], [av, hv], t1, t2, x0)
    #print(c1, c2, c3, c4, c5, f, f_squared, n, z1, z2, z3, " ALL RESULTS")

    #result = heights.pytorch_optimize_height(x0, vargs, term = termination)

    #print(x0, " initial")

    params = []
    error_array = []

    X = []
    Y = []
    Z = []
    V = []
    
    for i in range(10000):

        x0 = np.random.normal(1.6, 1e-8*i, no_points - 1)

        #x0 = x0 - np.mean(x0) + 1.6

        #print(np.mean(x0), "np mean")

        result, err = heights.pytorch_average_plane_optimize_height(x0, vargs, term = termination)

        params.append(np.mean(np.abs(np.array(result) - np.array(h))))
        print(err, np.mean(np.abs(np.array(result) - np.array(h))), " resultsss")
        error_array.append(err)

        X.append(np.abs(x0[0] - h[0]))
        Y.append(np.abs(x0[1] - h[1]))
        Z.append(np.abs(x0[2] - h[2]))
        V.append(err)

    ax0.set_yscale('log')
    ax0.set_xscale('log')
    ax0.scatter(params, error_array)
    fig.savefig('./plots/run_' + name + "/" + 'pdf_curve.png')

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, projection='3d')
    ax4.view_init(45,60)

    # here we create the surface plot, but pass V through a colormap
    # to create a different color for each patch
    
    #ax4.plot_surface(np.array(X), np.array(Y), np.array(Z), facecolors=cm.Oranges(V))

    img = ax4.scatter(X, Y, Z, c=V, cmap=plt.hot())
    fig4.colorbar(img)

    ax4.set_xlabel('$X$', fontsize=20)
    ax4.set_ylabel('$Y$', fontsize=20)
    ax4.set_zlabel('$Z$', fontsize=20)

    plt.show()

    plt.close('all')
    #result = heights.pytorch_average_optimize_height(h, vargs, term = termination)
    #result = scipy.optimize.minimize(heights.function_optimize_height, x0, vargs)
    '''
    for r in range(len(result)):
        print(result[r]['params'], " RESULT")

    print(h, " HEIGHT GROUND TRUTH")
    '''
    #heights.optimize_height(au, hu, av, hv, t1, t2)