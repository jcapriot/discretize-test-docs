# We create a simple mesh and refine the TreeMesh such that all cells that
# intersect the boxes are at the given levels.

import discretize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
tree_mesh = discretize.TreeMesh([32, 32])
tree_mesh.max_level
# 5

# Next we define the origins and furthest corners of the two rectangles, as
# well as the level we want to refine them to, and refine the mesh.

x0s = [[0.1, 0.1], [0.8, 0.8]]
x1s = [[0.3, 0.2], [0.9, 1.0]]
levels = [4, 5]
tree_mesh.refine_box(x0s, x1s, levels)

# Now lets look at the mesh, and overlay the boxes on it to ensure it refined
# where we wanted it to.

ax = tree_mesh.plot_grid()
rect = patches.Rectangle([0.1, 0.1], 0.2, 0.1, facecolor='none', edgecolor='r', linewidth=3)
ax.add_patch(rect)
rect = patches.Rectangle([0.8, 0.8], 0.1, 0.2, facecolor='none', edgecolor='k', linewidth=3)
ax.add_patch(rect)
plt.show()
