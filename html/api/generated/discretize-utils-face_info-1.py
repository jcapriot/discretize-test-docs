# Here we define a set of vertices for a tensor mesh. We then
# index 4 vertices for an irregular quadrilateral. The
# function *face_info* is used to compute the normal vector
# and the surface area.

from discretize.utils import face_info
from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({"font.size": 14})

# Define Corners of a uniform cube.

h = [1, 1]
mesh = TensorMesh([h, h, h])
xyz = mesh.nodes

# Choose the face indices,

A = np.array([0])
B = np.array([4])
C = np.array([26])
D = np.array([18])

# Compute average surface normal vector (normalized),

nvec, area = face_info(xyz, A, B, C, D)
nvec, area
# (array([[-0.70710678,  0.70710678,  0.        ]]), array([4.24264069]))

fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection='3d')
mesh.plot_grid(ax=ax)
k = [0, 4, 26, 18, 0]
xyz_quad = xyz[k, :]
ax.plot(xyz_quad[:, 0], xyz_quad[:, 1], xyz_quad[:, 2], 'r')
ax.text(-0.25, 0., 3., 'Area of the surface: {:g} $m^2$'.format(area[0]))
ax.text(-0.25, 0., 2.8, 'Normal vector: ({:.2f}, {:.2f}, {:.2f})'.format(
    nvec[0, 0], nvec[0, 1], nvec[0, 2])
)
plt.show()

# In our second example, the vertices are unable to define a flat
# surface in 3D space. However, we will demonstrate the *face_info*
# returns the average normal vector and an approximate surface area.

# Define the face indicies
A = np.array([0])
B = np.array([5])
C = np.array([26])
D = np.array([18])

# Compute average surface normal vector

nvec, area = face_info(xyz, A, B, C, D)
nvec, area
# (array([[-0.4472136 ,  0.89442719,  0.        ]]), array([2.23606798]))

fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection='3d')
mesh.plot_grid(ax=ax)
k = [0, 5, 26, 18, 0]
xyz_quad = xyz[k, :]
ax.plot(xyz_quad[:, 0], xyz_quad[:, 1], xyz_quad[:, 2], 'g')
ax.text(-0.25, 0., 3., 'Area of the surface: {:g} $m^2$'.format(area[0]))
ax.text(-0.25, 0., 2.8, 'Average normal vector: ({:.2f}, {:.2f}, {:.2f})'.format(
    nvec[0, 0], nvec[0, 1], nvec[0, 2])
)
plt.show()
