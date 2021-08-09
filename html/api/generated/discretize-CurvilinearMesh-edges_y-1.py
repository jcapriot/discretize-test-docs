# Here, we provide an example of a minimally staggered curvilinear mesh.
# In this case, the y-edges are primarily oriented along the y-direction.

from discretize import CurvilinearMesh
from discretize.utils import example_curvilinear_grid, mkvc
from matplotlib import pyplot as plt

x, y = example_curvilinear_grid([10, 10], "rotate")
mesh1 = CurvilinearMesh([x, y])
y_edges = mesh1.edges_y

fig1 = plt.figure(figsize=(5, 5))
ax1 = fig1.add_subplot(111)
mesh1.plot_grid(ax=ax1)
ax1.scatter(y_edges[:, 0], y_edges[:, 1], 30, 'r')
ax1.legend(['Mesh', 'Y-edges'], fontsize=16)
plt.plot()

# Here, we provide an example of a highly irregular curvilinear mesh.
# In this case, the y-edges are not aligned primarily along
# a particular direction.

x, y = example_curvilinear_grid([10, 10], "sphere")
mesh2 = CurvilinearMesh([x, y])
y_edges = mesh2.edges_y

fig2 = plt.figure(figsize=(5, 5))
ax2 = fig2.add_subplot(111)
mesh2.plot_grid(ax=ax2)
ax2.scatter(y_edges[:, 0], y_edges[:, 1], 30, 'r')
ax2.legend(['Mesh', 'X-edges'], fontsize=16)
plt.plot()
