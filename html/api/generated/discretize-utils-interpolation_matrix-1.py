# Here is a 1D example where a function evaluated on a regularly spaced grid
# is interpolated to a set of random locations. To compare the accuracy, the
# function is evaluated at the set of random locations.

from discretize.utils import interpolation_matrix
from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(14)

# Create an interpolation matrix

locs = np.random.rand(50)*0.8+0.1
x = np.linspace(0, 1, 7)
dense = np.linspace(0, 1, 200)
fun = lambda x: np.cos(2*np.pi*x)
Q = interpolation_matrix(locs, x)

fig1 = plt.figure(figsize=(5, 3))
ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(dense, fun(dense), 'k:', lw=3)
ax.plot(x, fun(x), 'ks', markersize=8)
ax.plot(locs, Q*fun(x), 'go', markersize=4)
ax.plot(locs, fun(locs), 'rs', markersize=4)
ax.legend(
    [
        'True Function',
        'True (discrete loc.)',
        'Interpolated (computed)',
        'True (interp. loc.)'
    ],
    loc='upper center'
)
plt.show()

# Here, demonstrate a similar example on a 2D mesh using a 2D Gaussian distribution.
# We interpolate the Gaussian from the nodes to cell centers and examine the relative
# error.

hx = np.ones(10)
hy = np.ones(10)
mesh = TensorMesh([hx, hy], x0='CC')
def fun(x, y):
    return np.exp(-(x**2 + y**2)/2**2)

# Define the the value at the mesh nodes,

nodes = mesh.nodes
val_nodes = fun(nodes[:, 0], nodes[:, 1])

centers = mesh.cell_centers
A = interpolation_matrix(
    centers, mesh.nodes_x, mesh.nodes_y
)
val_interp = A.dot(val_nodes)

# Plot the interpolated values, along with the true values at cell centers,

val_centers = fun(centers[:, 0], centers[:, 1])
fig = plt.figure(figsize=(11,3.3))
clim = (0., 1.)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
mesh.plot_image(val_centers, ax=ax1, clim=clim)
mesh.plot_image(val_interp, ax=ax2, clim=clim)
mesh.plot_image(val_centers-val_interp, ax=ax3, clim=clim)
ax1.set_title('Analytic at Centers')
ax2.set_title('Interpolated from Nodes')
ax3.set_title('Relative Error')
plt.show()
