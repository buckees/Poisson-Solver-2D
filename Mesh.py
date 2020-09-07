"""
Mesh file
"""

import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class MESHGRID(object):
    """Mesh object"""
    def __init__(self, width, height, nx, ny):
        self.width = width # domain width in x direction
        self.height = height # domain height in z direction
        self.nx = nx # nodes in x direction
        self.ny = ny # nodes in z direction
        self.delx = width/(nx-1) # delta x
        self.dely = height/(ny-1) # delta y
        self.posx = np.linspace(-width/2.0, width/2.0, nx)
        self.posy = np.linspace(-width/2.0, width/2.0, nx)
        
    def __str__(self):
        return """
               This mesh with domain of (%.3f m to %.3f m) in x 
                                        (%.3f m to %.3f m) in y
               with number of nodes in (nx, ny) = (%d, %d)
               """ \
                % (-self.width/2.0, self.width/2.0, 
                   -self.height/2.0, self.height/2.0, 
                   self.nx, self.ny)
    
    # assign input materials to the mesh
    def init_mesh(self):
        x = np.linspace(-self.width/2.0, self.width/2.0, self.nx)
        y = np.linspace(-self.height/2.0, self.height/2.0, self.nx)
        self.posx, self.posy = np.meshgrid(x, y)
    
    def plot_mesh(self):
        plt.plot(self.posx, self.posy, '.k')
    
    def calc_dist(self, point, I):
        sign = np.sign(I)
        x0, y0 = point
        distx = self.posx - x0
        disty = self.posy - y0
        dist = np.sqrt(np.power(distx, 2) + np.power(disty, 2))
        distx = np.divide(distx, dist, where=dist > 0.0,
                          out=np.zeros_like(dist))
        disty = np.divide(disty, dist, where=dist > 0.0,
                          out=np.zeros_like(dist))
        vecx, vecy = rotate(distx, disty, sign*pi/2.0)
        return dist, vecx, vecy

def rotate(vecx, vecy, angle):
    """
    Rotate a vector counterclockwise by a given angle
    The angle should be given in radians.
    """
    vecx_rot = cos(angle)*vecx - sin(angle)*vecy
    vecy_rot = sin(angle)*vecx + cos(angle)*vecy
    return vecx_rot, vecy_rot

if __name__ == '__main__':
    width, height, nx, ny = 1.0, 1.0, 11, 11
    mesh = MESHGRID(width, height, nx, ny)
    mesh.init_mesh()
    mesh.plot_mesh()
    dist, vecx, vecy = mesh.calc_dist((-0.3, -0.1))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(mesh.posx, mesh.posy, '.k')
    cs = ax.contour(mesh.posx, mesh.posy, dist)
#                 locator=ticker.LogLocator(subs=range(1,10)))
    ax.clabel(cs, cs.levels)
    q = ax.quiver(mesh.posx, mesh.posy, vecx, vecy)
#    q = ax.quiver(mesh.posx, mesh.posy, vecx, vecy)
#    ax.quiverkey(q, X=0.3, Y=1.1, U=10,
#                 label='Quiver key, length = 10', labelpos='E')


