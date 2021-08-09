# Here we compute the values of a scalar function on the nodes. We then create
# an averaging operator to approximate the function at the faces. We choose
# to define a scalar function that is strongly discontinuous in some places to
# demonstrate how the averaging operator will smooth out discontinuities.

# We start by importing the necessary packages and defining a mesh.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
h = np.ones(40)
mesh = TensorMesh([h, h], x0="CC")

# Then we, create a scalar variable on nodes

phi_n = np.zeros(mesh.nN)
xy = mesh.nodes
phi_n[(xy[:, 1] > 0)] = 25.0
phi_n[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

# Next, we construct the averaging operator and apply it to
# the discrete scalar quantity to approximate the value on the faces.

Anf = mesh.average_node_to_face
phi_f = Anf @ phi_n

# Plot the results,

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(121)
mesh.plot_image(phi_n, ax=ax1, v_type="N")
ax1.set_title("Variable at nodes")
ax2 = fig.add_subplot(122)
mesh.plot_image(phi_f, ax=ax2, v_type="F")
ax2.set_title("Averaged to faces")
plt.show()

# Below, we show a spy plot illustrating the sparsity and mapping
# of the operator

fig = plt.figure(figsize=(9, 9))
ax1 = fig.add_subplot(111)
ax1.spy(Anf, ms=1)
ax1.set_title("Node Index", fontsize=12, pad=5)
ax1.set_ylabel("Face Index", fontsize=12)
plt.show()
