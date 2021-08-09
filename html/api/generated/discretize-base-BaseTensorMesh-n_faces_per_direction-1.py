import discretize
import numpy as np
import matplotlib.pyplot as plt
M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
M.plot_grid(faces=True)
plt.show()
