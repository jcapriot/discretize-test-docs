# Here we compute the values of a vector function discretized to cell centers.
# We then create an averaging operator to approximate the function on the faces.

# We start by importing the necessary packages and defining a mesh.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
h = 0.5 * np.ones(40)
mesh = TensorMesh([h, h], x0="CC")

# Then we create a discrete vector at cell centers,

centers = mesh.cell_centers
u_x = -(centers[:, 1] / np.sqrt(np.sum(centers ** 2, axis=1))) * np.exp(
    -(centers[:, 0] ** 2 + centers[:, 1] ** 2) / 6 ** 2
)
u_y = (centers[:, 0] / np.sqrt(np.sum(centers ** 2, axis=1))) * np.exp(
    -(centers[:, 0] ** 2 + centers[:, 1] ** 2) / 6 ** 2
)
u_c = np.r_[u_x, u_y]

# Next, we construct the averaging operator and apply it to
# the discrete vector quantity to approximate the value on the faces.

Acf = mesh.average_cell_vector_to_face
u_f = Acf @ u_c

# And plot the results,

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(121)
mesh.plot_image(u_c, ax=ax1, v_type="CCv", view='vec')
ax1.set_title("Variable at faces", fontsize=16)
ax2 = fig.add_subplot(122)
mesh.plot_image(u_f, ax=ax2, v_type="F", view='vec')
ax2.set_title("Averaged to cell centers", fontsize=16)
plt.show()

# Below, we show a spy plot illustrating the sparsity and mapping
# of the operator

fig = plt.figure(figsize=(9, 9))
ax1 = fig.add_subplot(111)
ax1.spy(Acf, ms=1)
ax1.set_title("Cell Vector Index", fontsize=12, pad=5)
ax1.set_ylabel("Face Index", fontsize=12)
plt.show()
