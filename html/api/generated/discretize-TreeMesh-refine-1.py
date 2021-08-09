# Here, we define a QuadTree mesh with a domain width of 1 along
# the x and y axes. We then refine the mesh to its maximum level
# at locations within a distance of 0.2 of point (0.5, 0.5). The
# function accepts and instance of :class:`~discretize.tree_mesh.TreeCell`
# and returns an integer value denoting its level of refinement.

from discretize import TreeMesh
from matplotlib import pyplot

# Define a mesh and refine it radially outward using the custom defined function

mesh = TreeMesh([32,32])
def func(cell):
    r = np.linalg.norm(cell.center-0.5)
    return mesh.max_level if r<0.2 else mesh.max_level-2
mesh.refine(func)

mesh.plot_grid()
pyplot.show()
