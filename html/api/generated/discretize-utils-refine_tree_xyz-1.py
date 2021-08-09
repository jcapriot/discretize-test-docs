# Here we use the **refine_tree_xyz** function refine a tree mesh
# based on topography as well as a cluster of points.

from discretize import TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
import matplotlib.pyplot as plt
import numpy as np

dx = 5  # minimum cell width (base mesh cell width) in x
dy = 5  # minimum cell width (base mesh cell width) in y
x_length = 300.0  # domain width in x
y_length = 300.0  # domain width in y

# Compute number of base mesh cells required in x and y

nbcx = 2 ** int(np.round(np.log(x_length / dx) / np.log(2.0)))
nbcy = 2 ** int(np.round(np.log(y_length / dy) / np.log(2.0)))

# Define the base mesh

hx = [(dx, nbcx)]
hy = [(dy, nbcy)]
mesh = TreeMesh([hx, hy], x0="CC")

# Refine surface topography

xx = mesh.vectorNx
yy = -3 * np.exp((xx ** 2) / 100 ** 2) + 50.0
pts = np.c_[mkvc(xx), mkvc(yy)]
mesh = refine_tree_xyz(
    mesh, pts, octree_levels=[2, 4], method="surface", finalize=False
)

# Refine mesh near points

xx = np.array([-10.0, 10.0, 10.0, -10.0])
yy = np.array([-40.0, -40.0, -60.0, -60.0])
pts = np.c_[mkvc(xx), mkvc(yy)]
mesh = refine_tree_xyz(
    mesh, pts, octree_levels=[4, 2], method="radial", finalize=True
)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
mesh.plotGrid(ax=ax)
ax.set_xbound(mesh.x0[0], mesh.x0[0] + np.sum(mesh.hx))
ax.set_ybound(mesh.x0[1], mesh.x0[1] + np.sum(mesh.hy))
ax.set_title("QuadTree Mesh")
plt.show()
