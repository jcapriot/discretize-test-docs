# Here, we define a 2D tensor mesh that has both a core region and padding.
# We use the function **extract_core_mesh** to return a mesh which contains
# only the core region.

from discretize.utils import extract_core_mesh
from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({"font.size": 14})

# Form a mesh of a uniform cube
h = [(1., 5, -1.5), (1., 20), (1., 5, 1.5)]
mesh = TensorMesh([h, h], origin='CC')

# Plot original mesh

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)
mesh.plot_grid(ax=ax)
ax.set_title('Original Tensor Mesh')
plt.show()

# Set the limits for the cutoff of the core mesh (dim, 2)

xlim = np.c_[-10., 10]
ylim = np.c_[-10., 10]
core_limits = np.r_[xlim, ylim]

# Extract indices of core mesh cells and the core mesh, then plot

core_ind, core_mesh = extract_core_mesh(core_limits, mesh)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111)
core_mesh.plot_grid(ax=ax)
ax.set_title('Core Mesh')
plt.show()
