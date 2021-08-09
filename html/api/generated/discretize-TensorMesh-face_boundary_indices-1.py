# Here, we construct a 4 by 3 cell 2D tensor mesh and return the indices
# of the x and y-boundary faces. In this case there are 3 x-faces on each
# x-boundary, and there are 4 y-faces on each y-boundary.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt

hx = [1, 1, 1, 1]
hy = [2, 2, 2]
mesh = TensorMesh([hx, hy])
ind_Bx1, ind_Bx2, ind_By1, ind_By2 = mesh.face_boundary_indices

ax = plt.subplot(111)
mesh.plot_grid(ax=ax)
ax.scatter(*mesh.faces_x[ind_Bx1].T)
plt.show()
