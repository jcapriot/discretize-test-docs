# Using the :py:func:`~discretize.utils.example_curvilinear_grid` utility,
# we provide an example of a curvilinear mesh.

from discretize import CurvilinearMesh
from discretize.utils import example_curvilinear_grid
import matplotlib.pyplot as plt

# The example grid slightly rotates the nodes in the center of the mesh,

x, y = example_curvilinear_grid([10, 10], "rotate")
x.shape
# (11, 11)
y.shape
# (11, 11)
curvilinear_mesh = CurvilinearMesh([x, y])
curvilinear_mesh.shape_nodes
# (11, 11)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
curvilinear_mesh.plot_grid(ax=ax)
plt.show()
