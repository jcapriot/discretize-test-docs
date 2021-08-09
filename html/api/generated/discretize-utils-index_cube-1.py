# Here, we construct a small 2D tensor mesh
# (works for a curvilinear mesh as well) and use *index_cube*
# to find the indices of the 'A' and 'C' nodes. We then
# plot the mesh, as well as the 'A' and 'C' node locations.

from discretize import TensorMesh
from discretize.utils import index_cube
from matplotlib import pyplot as plt
import numpy as np

# Create a simple tensor mesh.

n_cells = 5
h = 2*np.ones(n_cells)
mesh = TensorMesh([h, h], x0='00')

# Get indices of 'A' and 'C' nodes for all cells.

A, C = index_cube('AC', [n_cells+1, n_cells+1])

# Plot mesh and the locations of the A and C nodes

fig1 = plt.figure(figsize=(5, 5))
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
mesh.plot_grid(ax=ax1)
ax1.scatter(mesh.nodes[A, 0], mesh.nodes[A, 1], 100, 'r', marker='^')
ax1.scatter(mesh.nodes[C, 0], mesh.nodes[C, 1], 100, 'g', marker='v')
ax1.set_title('A nodes (red) and C nodes (green)')
plt.show()
