# Below, we demonstrate how to apply the face-x divergence operator,
# the mapping of the face-x divergence operator and its sparsity.
# Our example is carried out on a 2D mesh but it can
# be done equivalently for a 3D mesh.

# We start by importing the necessary packages and modules.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# For a discrete scalar quantity :math:`\boldsymbol{\phi}` defined on the
# x-faces, we take the x-derivative by constructing the face-x divergence
# operator and multiplying as a matrix-vector product.

h = np.ones(40)
mesh = TensorMesh([h, h], "CC")

# Create a discrete quantity on x-faces

faces_x = mesh.faces_x
phi = np.exp(-(faces_x[:, 0] ** 2) / 8** 2)

# Construct the x-divergence operator and apply to vector

Dx = mesh.face_x_divergence
dphi_dx = Dx @ phi

fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121)
w = np.r_[phi, np.ones(mesh.nFy)]  # Need vector on all faces for image plot
mesh.plot_image(w, ax=ax1, v_type="Fx")
ax1.set_title("Scalar on x-faces", fontsize=14)
ax2 = fig.add_subplot(122)
mesh.plot_image(dphi_dx, ax=ax2)
ax2.set_yticks([])
ax2.set_ylabel("")
ax2.set_title("X-derivative at cell center", fontsize=14)
plt.show()

# The discrete x-face divergence operator is a sparse matrix that maps
# from x-faces to cell centers. To demonstrate this, we construct
# a small 2D mesh. We then show the ordering of the elements in
# the original discrete quantity :math:`\boldsymbol{\phi}}` and its
# x-derivative :math:`\partial \boldsymbol{\phi}}/ \partial x` as well as a
# spy plot.

mesh = TensorMesh([[(1, 6)], [(1, 3)]])
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(211)
mesh.plot_grid(ax=ax1)
ax1.plot(
    mesh.faces_x[:, 0], mesh.faces_x[:, 1], "g>", markersize=8
)
for ii, loc in zip(range(mesh.nFx), mesh.faces_x):
    ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="g")
ax1.plot(
    mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8
)
for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
    ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="r")

ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['bottom'].set_color('white')
ax1.spines['top'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.spines['right'].set_color('white')
ax1.set_xlabel('X', fontsize=16, labelpad=-5)
ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
ax1.set_title("Mapping of Face-X Divergence", fontsize=14, pad=15)
ax1.legend(
    ['Mesh', '$\mathbf{\phi}$ (x-faces)', '$\partial \mathbf{phi}/\partial x$ (centers)'],
    loc='upper right', fontsize=14
)
ax2 = fig.add_subplot(212)
ax2.spy(mesh.face_x_divergence)
ax2.set_title("Spy Plot", fontsize=14, pad=5)
ax2.set_ylabel("Cell Index", fontsize=12)
ax2.set_xlabel("X-Face Index", fontsize=12)
plt.show()
