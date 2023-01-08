import numpy as np
from matplotlib import pyplot as plt
from numpy import random

#Most of this code is reused from run_simulation.py so look at that to see how the code works
axs=[]

no_points = 4
random.seed(43)
f = random.uniform(2000, 5000)
t1 = 940.0
t2 = 560.0
h1 = 1.6
h2 = 100.6
h3 = 1.6

h = [h1,h2,h3]
n = [random.uniform(-5, 5), random.uniform(1, 11), random.uniform(-5, 5)]
n = n / np.linalg.norm(n)
p = [random.uniform(-5, 5), random.uniform(-5, 5) - 10, random.uniform(-5, 5)]

c = np.array([[f, 0, t1], [0, f, t2], [0, 0, 1]])
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
    hx.append(([ax[i], ay[i], az[i]]+h[i]*n)[0])
    hy.append(([ax[i], ay[i], az[i]]+h[i]*n)[1])
    hz.append(([ax[i], ay[i], az[i]]+h[i]*n)[2])
    sub.append([hx[i]-ax[i], hy[i]-ay[i], hz[i]-az[i]])

#manually construct the projective matrix
for i in range(0, no_points-1):
    au.append(np.dot(c, np.array([[ax[i]], [ay[i]], [az[i]]]))[0].item())
    av.append(np.dot(c, np.array([[ax[i]], [ay[i]], [az[i]]]))[1].item())
    aw.append(np.dot(c, np.array([[ax[i]], [ay[i]], [az[i]]]))[2].item())
    au[i] = au[i]/aw[i]
    av[i] = av[i]/aw[i]
    aw[i] = 1


# This renders the 3D scene
fig3d = plt.figure()
fig3d.set_size_inches((5, 5))
plt3d = fig3d.gca(projection='3d')

plt3d.plot([ax[0],hx[0]],[ay[0],hy[0]], [az[0],hz[0]],color = 'black')
plt3d.plot([ax[1],hx[1]],[ay[1],hy[1]], [az[1],hz[1]],color = 'red')
plt3d.plot([ax[2],hx[2]],[ay[2],hy[2]], [az[2],hz[2]], color = 'green')

plt3d.plot([p[0]], [p[1]], [p[2]], marker='o', markersize=3, color="red")
plt3d.plot([0], [0], [0], marker='o', markersize=3, color="blue")

#pointHead = p + np.array([n[0], n[1], n[2]])*h

xx, z = np.meshgrid(range(-20,20), range(-20,20))
yy =  p[1] + (n[0]*(p[0] - xx) + n[2]*(p[2] - z))/n[1]

x, z_orig = np.meshgrid(range(-20,20), range(-20,20))
y = p[1] + (n[0]*(p[0] - x) + n[2]*(p[2] - z_orig))/n[1]

plt3d.set_xlim3d(-20, 20)
plt3d.set_ylim3d(-20, 20)
plt3d.set_zlim3d(-20, 20)

#plt3d.plot_surface(xx, yy, z)
plt3d.plot_surface(x, y, z_orig)

plt3d.plot([2 , 0], [0 , 0], [0, 0], color="blue")
plt3d.plot([0 , 0], [2 , 0], [0, 0], color="red")
plt3d.plot([0 , 0], [0 , 0], [2, 0], color="green")

plt.grid()
plt.show()

