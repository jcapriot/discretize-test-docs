# We create a simple mesh and refine the tree mesh such that all cells that
# intersect the spherical balls are at the given levels.

import discretize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
tree_mesh = discretize.TreeMesh([32, 32])
tree_mesh.max_level
# 5

# Next we define the center and radius of the two spheres, as well as the level
# we want to refine them to, and refine the mesh.

centers = [[0.1, 0.3], [0.6, 0.8]]
radii = [0.2, 0.3]
levels = [4, 5]
tree_mesh.refine_ball(centers, radii, levels)

# Now lets look at the mesh, and overlay the balls on it to ensure it refined
# where we wanted it to.

ax = tree_mesh.plot_grid()
circ = patches.Circle(centers[0], radii[0], facecolor='none', edgecolor='r', linewidth=3)
ax.add_patch(circ)
circ = patches.Circle(centers[1], radii[1], facecolor='none', edgecolor='k', linewidth=3)
ax.add_patch(circ)
plt.show()
