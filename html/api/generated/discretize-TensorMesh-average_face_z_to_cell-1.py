# Here we compute the values of a scalar function on the z-faces. We then create
# an averaging operator to approximate the function at cell centers. We choose
# to define a scalar function that is strongly discontinuous in some places to
# demonstrate how the averaging operator will smooth out discontinuities.

# We start by importing the necessary packages and defining a mesh.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt

h = np.ones(40)
mesh = TensorMesh([h, h, h], x0="CCC")

# Create a scalar variable on z-faces

phi_z = np.zeros(mesh.nFz)
xyz = mesh.faces_z
phi_z[(xyz[:, 2] > 0)] = 25.0
phi_z[(xyz[:, 2] < -10.0) & (xyz[:, 0] > -10.0) & (xyz[:, 0] < 10.0)] = 50.0

# Next, we construct the averaging operator and apply it to
# the discrete scalar quantity to approximate the value at cell centers.
# We plot the original scalar and its average at cell centers for a
# slice at y=0.

Azc = mesh.average_face_z_to_cell
phi_c = Azc @ phi_z

# And plot the results,

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(121)
v = np.r_[np.zeros(mesh.nFx+mesh.nFy), phi_z]  # create vector for plotting
mesh.plot_slice(v, ax=ax1, normal='Y', slice_loc=0, v_type="Fz")
ax1.set_title("Variable at z-faces", fontsize=16)
ax2 = fig.add_subplot(122)
mesh.plot_image(phi_c, ax=ax2, normal='Y', slice_loc=0, v_type="CC")
ax2.set_title("Averaged to cell centers", fontsize=16)
plt.show()

# Below, we show a spy plot illustrating the sparsity and mapping
# of the operator

fig = plt.figure(figsize=(9, 9))
ax1 = fig.add_subplot(111)
ax1.spy(Azc, ms=1)
ax1.set_title("Z-Face Index", fontsize=12, pad=5)
ax1.set_ylabel("Cell Index", fontsize=12)
plt.show()
