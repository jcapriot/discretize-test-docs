# Plotting a 2D TensorMesh grid

from matplotlib import pyplot as plt
import discretize
import numpy as np
h1 = np.linspace(.1, .5, 3)
h2 = np.linspace(.1, .5, 5)
mesh = discretize.TensorMesh([h1, h2])
mesh.plot_grid(nodes=True, faces=True, centers=True, lines=True)
plt.show()

# Plotting a 3D TensorMesh grid

from matplotlib import pyplot as plt
import discretize
import numpy as np
h1 = np.linspace(.1, .5, 3)
h2 = np.linspace(.1, .5, 5)
h3 = np.linspace(.1, .5, 3)
mesh = discretize.TensorMesh([h1, h2, h3])
mesh.plot_grid(nodes=True, faces=True, centers=True, lines=True)
plt.show()

# Plotting a 2D CurvilinearMesh

from matplotlib import pyplot as plt
import discretize
X, Y = discretize.utils.exampleLrmGrid([10, 10], 'rotate')
M = discretize.CurvilinearMesh([X, Y])
M.plot_grid()
plt.show()

# Plotting a 3D CurvilinearMesh

from matplotlib import pyplot as plt
import discretize
X, Y, Z = discretize.utils.exampleLrmGrid([5, 5, 5], 'rotate')
M = discretize.CurvilinearMesh([X, Y, Z])
M.plot_grid()
plt.show()

# Plotting a 2D TreeMesh

from matplotlib import pyplot as plt
import discretize
M = discretize.TreeMesh([32, 32])
M.insert_cells([[0.25, 0.25]], [4])
M.plot_grid()
plt.show()

# Plotting a 3D TreeMesh

from matplotlib import pyplot as plt
import discretize
M = discretize.TreeMesh([32, 32, 32])
M.insert_cells([[0.3, 0.75, 0.22]], [4])
M.plot_grid()
plt.show()
