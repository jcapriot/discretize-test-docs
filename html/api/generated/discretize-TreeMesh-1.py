# Here we generate a basic 2D tree mesh.

from discretize import TreeMesh
import numpy as np
import matplotlib.pyplot as plt

# Define base mesh (domain and finest discretization),

dh = 5    # minimum cell width (base mesh cell width)
nbc = 64  # number of base mesh cells
h = dh * np.ones(nbc)
mesh = TreeMesh([h, h])

# Define corner points for a rectangular box, and subdived the mesh within the box
# to the maximum refinement level.

x0s = [120.0, 80.0]
x1s = [240.0, 160.0]
levels = [mesh.max_level]
mesh.refine_box(x0s, x1s, levels)

mesh.plot_grid()
plt.show()
