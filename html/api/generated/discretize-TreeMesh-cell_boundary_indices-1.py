# Here, we construct a small 2D tree mesh and return the indices
# of the x and y-boundary cells.

from discretize import TreeMesh
from discretize.utils import refine_tree_xyz
import numpy as np
import matplotlib.pyplot as plt

hx = np.ones(16)
hy = np.ones(16)
mesh = TreeMesh([hx, hy])
mesh.refine_ball([4.0,4.0], [4.0], [4])
ind_Bx1, ind_Bx2, ind_By1, ind_By2 = mesh.cell_boundary_indices

ax = plt.subplot(111)
mesh.plot_grid(ax=ax)
ax.scatter(*mesh.cell_centers[ind_Bx1].T)
plt.show()
