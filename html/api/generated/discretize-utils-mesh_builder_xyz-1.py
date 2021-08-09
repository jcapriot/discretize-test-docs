import discretize
import matplotlib.pyplot as plt
import numpy as np

xyLoc = np.random.randn(8,2)
mesh = discretize.utils.mesh_builder_xyz(
    xyLoc, [0.1, 0.1], depth_core=0.5,
    padding_distance=[[1,2], [1,0]],
    mesh_type='tensor',
)

axs = plt.subplot()
mesh.plot_image(mesh.cell_volumes, grid=True, ax=axs)
axs.scatter(xyLoc[:,0], xyLoc[:,1], 15, c='w', zorder=3)
axs.set_aspect('equal')
plt.show()
