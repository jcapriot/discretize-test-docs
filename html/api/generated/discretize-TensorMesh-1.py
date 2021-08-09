# An example of a 2D tensor mesh is shown below. Here we use a list of tuple to
# define the discretization along the x-axis and a numpy array to define the
# discretization along the y-axis. We also use a string argument to center the
# x-axis about x = 0 and set the top of the mesh to y = 0.

from discretize import TensorMesh
import matplotlib.pyplot as plt

ncx = 10      # number of core mesh cells in x
dx = 5        # base cell width x
npad_x = 3    # number of padding cells in x
exp_x = 1.25  # expansion rate of padding cells in x
ncy = 24      # total number of mesh cells in y
dy = 5        # base cell width y

hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
hy = dy * np.ones(ncy)
mesh = TensorMesh([hx, hy], origin='CN')

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
mesh.plot_grid(ax=ax)
plt.show()
