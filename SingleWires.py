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
width, height, nx, ny = 10.0, 10.0, 51, 51
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
    
b[0+(nx-2)*int(ny/2-1)+int(nx/2-1)] = MU0*I/(2.0*pi)
#b[-1-(nx-2)*int(ny/2-1)-30] = MU0*I/(2.0*pi)
 
phi=scipy.linalg.solve(A,b)
phi = phi.reshape((nx-2, ny-2))
phi = np.pad(phi, pad_width=1, mode='constant', constant_values=0)

def curl(A):
    """
    calculate the curl of A, which is a matrix of Fz in z direction
    Boundary of A is set to zero
    Using center difference, only A[1:-2, 1:-2] is calculated
    """
    U = np.zeros_like(A)
    V = np.zeros_like(A)
    bf = np.zeros_like(A)
    nx, ny = A.shape
    for i in range(1, nx-2):
        for j in range(1, ny-2):
            U[i,j] = (A[i, j+1] - A[i, j-1])/mesh.delx/2.0
            V[i,j] = -(A[i+1, j] - A[i-1, j])/mesh.dely/2.0
    bf = np.sqrt(np.power(U, 2) + np.power(V, 2))
    U = np.divide(U, bf, where=bf>0, out=np.zeros_like(bf))
    V = np.divide(V, bf, where=bf>0, out=np.zeros_like(bf))
    return U, V, bf

U, V, bf = curl(phi)

phi = abs(phi)
print('B field min = %.2e max = %.2e' % (phi.min(), phi.max()))

#
fig, ax = plt.subplots(1, 2, figsize=(6,3))
# and the norm:
lev_exp = np.arange(np.floor(np.log10(phi[np.nonzero(phi)].min())),
                    np.ceil(np.log10(phi.max())), 0.1)
levs = np.power(10, lev_exp)
cs = ax[0].contour(mesh.posx, mesh.posy,
                   phi, levs, norm=colors.LogNorm())

lev_exp = np.arange(np.floor(np.log10(bf[np.nonzero(bf)].min())),
                    np.ceil(np.log10(bf.max())), 0.1)
levs = np.power(10, lev_exp)
cs = ax[1].contour(mesh.posx, mesh.posy, bf, levs, norm=colors.LogNorm())

fig, ax = plt.subplots(1, 1, figsize=(6,6))
ax.streamplot(mesh.posx, mesh.posy, V, U)
