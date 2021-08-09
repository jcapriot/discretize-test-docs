# 2D ``TensorMesh`` plotting

from matplotlib import pyplot as plt
import discretize
import numpy as np
M = discretize.TensorMesh([20, 20])
v = np.sin(M.gridCC[:, 0]*2*np.pi)*np.sin(M.gridCC[:, 1]*2*np.pi)
M.plot_image(v)
plt.show()

# 3D ``TensorMesh`` plotting

import discretize
import numpy as np
M = discretize.TensorMesh([20, 20, 20])
v = np.sin(M.gridCC[:, 0]*2*np.pi)*np.sin(M.gridCC[:, 1]*2*np.pi)*np.sin(M.gridCC[:, 2]*2*np.pi)
M.plot_image(v, annotation_color='k')
plt.show()
