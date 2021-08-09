# Here we compute the values of a vector function discretized to the edges.
# We then create an averaging operator to approximate the function at cell centers.

# We start by importing the necessary packages and defining a mesh.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
h = 0.5 * np.ones(40)
mesh = TensorMesh([h, h], x0="CC")

# Then we create a discrete vector on mesh edges

edges_x = mesh.edges_x
edges_y = mesh.edges_y
u_ex = -(edges_x[:, 1] / np.sqrt(np.sum(edges_x ** 2, axis=1))) * np.exp(
    -(edges_x[:, 0] ** 2 + edges_x[:, 1] ** 2) / 6 ** 2
)
u_ey = (edges_y[:, 0] / np.sqrt(np.sum(edges_y ** 2, axis=1))) * np.exp(
    -(edges_y[:, 0] ** 2 + edges_y[:, 1] ** 2) / 6 ** 2
)
u_e = np.r_[u_ex, u_ey]

# Next, we construct the averaging operator and apply it to
# the discrete vector quantity to approximate the value at cell centers.

Aec = mesh.average_edge_to_cell_vector
u_c = Aec @ u_e

# And plot the results,

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(121)
mesh.plot_image(u_e, ax=ax1, v_type="E", view='vec')
ax1.set_title("Variable at edges", fontsize=16)
ax2 = fig.add_subplot(122)
mesh.plot_image(u_c, ax=ax2, v_type="CCv", view='vec')
ax2.set_title("Averaged to cell centers", fontsize=16)
plt.show()

# Below, we show a spy plot illustrating the sparsity and mapping
# of the operator

fig = plt.figure(figsize=(9, 9))
ax1 = fig.add_subplot(111)
ax1.spy(Aec, ms=1)
ax1.set_title("Edge Index", fontsize=12, pad=5)
ax1.set_ylabel("Cell Vector Index", fontsize=12)
plt.show()
