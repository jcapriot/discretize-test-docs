# Create two meshes with the same extent, but different divisions (the meshes
# do not have to be the same extent).

import numpy as np
from discretize import TensorMesh
h1 = np.ones(32)
h2 = np.ones(16)*2
mesh_in = TensorMesh([h1, h1])
mesh_out = TensorMesh([h2, h2])

# Create a random model defined on the input mesh, and use volume averaging to
# interpolate it to the output mesh.

from discretize.utils import volume_average
model1 = np.random.rand(mesh_in.nC)
model2 = volume_average(mesh_in, mesh_out, model1)

# Because these two meshes' cells are perfectly aligned, but the output mesh
# has 1 cell for each 4 of the input cells, this operation should effectively
# look like averaging each of those cells values

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3))
ax1 = plt.subplot(121)
mesh_in.plot_image(model1, ax=ax1)
ax2 = plt.subplot(122)
mesh_out.plot_image(model2, ax=ax2)
plt.show()
