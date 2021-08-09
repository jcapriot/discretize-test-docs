# Here, we generate a random model for a 2D tensor mesh and plot.

from discretize import TensorMesh
from discretize.utils import random_model, mkvc
import matplotlib as mpl
import matplotlib.pyplot as plt

h = [(1., 50)]
vmin, vmax = 0., 1.
mesh = TensorMesh([h, h])

model = random_model(mesh.shape_cells, seed=4, bounds=[vmin, vmax])

fig = plt.figure(figsize=(5, 4))
ax = plt.subplot(111)
im, = mesh.plot_image(model, grid=False, ax=ax, clim=[vmin, vmax])
cbar = plt.colorbar(im)
ax.set_title('Random Tensor Model')
plt.show()
