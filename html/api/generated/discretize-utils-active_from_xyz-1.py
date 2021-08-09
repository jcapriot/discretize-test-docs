# Here we define the active cells below a parabola. We demonstrate the differences
# that appear when using the 'CC' and 'N' options for *reference_grid*.

import matplotlib.pyplot as plt
import numpy as np
from discretize import TensorMesh
from discretize.utils import active_from_xyz

mesh = TensorMesh([5, 5])
topo_func = lambda x: -3*(x-0.2)*(x-0.8)+.5
topo_points = np.linspace(0, 1)
topo_vals = topo_func(topo_points)
active_cc = active_from_xyz(mesh, np.c_[topo_points, topo_vals], grid_reference='CC')
active_n = active_from_xyz(mesh, np.c_[topo_points, topo_vals], grid_reference='N')

ax = plt.subplot(121)
mesh.plot_image(active_cc, ax=ax)
mesh.plot_grid(centers=True, ax=ax)
ax.plot(np.linspace(0,1), topo_func(np.linspace(0,1)), color='C3')
ax.set_title("CC")
ax = plt.subplot(122)
mesh.plot_image(active_n, ax=ax)
mesh.plot_grid(nodes=True, ax=ax)
ax.plot(np.linspace(0,1), topo_func(np.linspace(0,1)), color='C3')
ax.set_title("N")
plt.show()
