# -*- coding: utf-8 -*-
"""
2D Poisson Solver 
"""

# src-ch7/laplace_Diriclhet1.py
import numpy as np
import scipy 
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
#import matplotlib; matplotlib.use('Qt4Agg')
import matplotlib.pylab as plt
from matplotlib import colors, ticker, cm
from math import pi

from Constants import MU0
from Mesh import MESHGRID


#set infinite-long wire to position (wx, wy) with infinitesimal radius
I = 1.0 # wire current in A
# According to Ampere's law in integral form
# B(r|r>r0) = mu0*I/(2*pi*r)
#The earth's magnetic field is about 0.5 gauss. 
width, height, nx, ny = 1.0, 1.0, 21, 21
mesh = MESHGRID(width, height, nx, ny)
mesh.init_mesh()


mn = (nx-2)*(ny-2)
b = np.zeros(mn) #RHS

#scalar broadcasting version:
A = scipy.sparse.diags([1, 1, -4, 1, 1],
                       [-(nx-2), -1, 0, 1, (ny-2)], 
                       shape=(mn, mn)).toarray()

# update A matrix
for i in range(1, nx-2):
    A[(nx-2)*i-1, (ny-2)*i] = 0
    A[(ny-2)*i, (nx-2)*i-1] = 0
    
print(A)
# update RHS:
#b[4], b[9], b[14] = -100, -100, -100
#b[int(mn/2)] = -MU0*I/(2.0*pi)
b[0] = -MU0*I/(2.0*pi)
b[-1] = MU0*I/(2.0*pi)
 
phi=scipy.linalg.solve(A,b)
phi = phi.reshape((nx-2, ny-2))
phi = np.pad(phi, pad_width=1, mode='constant', constant_values=0)

phi = abs(phi)
print('B field min = %.2e max = %.2e' % (phi.min(), phi.max()))

#
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(mesh.posx, mesh.posy, '.k')
# Alternatively, you can manually set the levels
# and the norm:
lev_exp = np.arange(np.floor(np.log10(phi[np.nonzero(phi)].min())),
                    np.ceil(np.log10(phi.max())), 0.1)
levs = np.power(10, lev_exp)
cs = ax.contour(mesh.posx, mesh.posy, phi, levs, norm=colors.LogNorm())
#ax.clabel(cs, cs.levels)
#fig.colorbar(cs)
#ax.quiver(mesh.posx, mesh.posy, vx, vy)
#ax.plot(pos1[0], pos1[1],
#        color='red', marker='o', markersize=15)
#ax.plot(pos2[0], pos2[1],
#        color='red', marker='o', markersize=15)
