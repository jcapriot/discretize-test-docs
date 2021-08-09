import discretize
import matplotlib.pyplot as plt
import numpy as np
M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
M.plot_grid(edges=True)
plt.show()
