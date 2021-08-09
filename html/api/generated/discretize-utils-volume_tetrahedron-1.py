# Here we define a small 3D tensor mesh. 4 nodes are chosen to
# be the verticies of a tetrahedron. We compute the volume of this
# tetrahedron. Note that xyz locations for the verticies can be
# scattered and do not require regular spacing.

from discretize.utils import volume_tetrahedron
from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({"font.size": 14})

# Define corners of a uniform cube

h = [1, 1]
mesh = TensorMesh([h, h, h])
xyz = mesh.nodes

# Specify the indicies of the corner points

A = np.array([0])
B = np.array([6])
C = np.array([8])
D = np.array([24])

# Compute volume for all tetrahedra and the extract first one

vol = volume_tetrahedron(xyz, A, B, C, D)
vol = vol[0]
vol
# array([1.33333333])

fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection='3d')
mesh.plot_grid(ax=ax)
k = [0, 6, 8, 0, 24, 6, 24, 8]
xyz_tetra = xyz[k, :]
ax.plot(xyz_tetra[:, 0], xyz_tetra[:, 1], xyz_tetra[:, 2], 'r')
ax.text(-0.25, 0., 3., 'Volume of the tetrahedron: {:g} $m^3$'.format(vol))
plt.show()
