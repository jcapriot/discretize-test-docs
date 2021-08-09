# Here we compute the values of a scalar function on the x-edges. We then create
# an averaging operator to approximate the function at cell centers. We choose
# to define a scalar function that is strongly discontinuous in some places to
# demonstrate how the averaging operator will smooth out discontinuities.

# We start by importing the necessary packages and defining a mesh.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
h = np.ones(40)
mesh = TensorMesh([h, h], x0="CC")

# Then we create a scalar variable on x-edges,

phi_x = np.zeros(mesh.nEx)
xy = mesh.edges_x
phi_x[(xy[:, 1] > 0)] = 25.0
phi_x[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

# Next, we construct the averaging operator and apply it to
# the discrete scalar quantity to approximate the value at cell centers.

Axc = mesh.average_edge_x_to_cell
phi_c = Axc @ phi_x

# And plot the results,

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(121)
v = np.r_[phi_x, np.zeros(mesh.nEy)]  # create vector for plotting function
mesh.plot_image(v, ax=ax1, v_type="Ex")
ax1.set_title("Variable at x-edges", fontsize=16)
ax2 = fig.add_subplot(122)
mesh.plot_image(phi_c, ax=ax2, v_type="CC")
ax2.set_title("Averaged to cell centers", fontsize=16)
plt.show()

# Below, we show a spy plot illustrating the sparsity and mapping
# of the operator

fig = plt.figure(figsize=(9, 9))
ax1 = fig.add_subplot(111)
ax1.spy(Axc, ms=1)
ax1.set_title("X-Edge Index", fontsize=12, pad=5)
ax1.set_ylabel("Cell Index", fontsize=12)
plt.show()
