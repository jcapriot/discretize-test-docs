# Below, we demonstrate how to apply the face divergence operator to
# a discrete vector, the mapping of the face divergence operator and
# its sparsity. Our example is carried out on a 2D mesh but it can
# be done equivalently for a 3D mesh.

# We start by importing the necessary packages and modules.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# For a discrete vector quantity defined on the faces, we take the
# divergence by constructing the divergence operator and multiplying
# as a matrix-vector product.

# Create a uniform grid

h = np.ones(20)
mesh = TensorMesh([h, h], "CC")

# Create a discrete vector on mesh faces

faces_x = mesh.faces_x
faces_y = mesh.faces_y
ux = (faces_x[:, 0] / np.sqrt(np.sum(faces_x ** 2, axis=1))) * np.exp(
    -(faces_x[:, 0] ** 2 + faces_x[:, 1] ** 2) / 6 ** 2
)
uy = (faces_y[:, 1] / np.sqrt(np.sum(faces_y ** 2, axis=1))) * np.exp(
    -(faces_y[:, 0] ** 2 + faces_y[:, 1] ** 2) / 6 ** 2
)
u = np.r_[ux, uy]

# Construct the divergence operator and apply to vector

Df = mesh.face_divergence
div_u = Df @ u

fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121)
mesh.plot_image(
    u, ax=ax1, v_type="F", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax1.set_title("Vector at cell faces", fontsize=14)
ax2 = fig.add_subplot(122)
mesh.plot_image(div_u, ax=ax2)
ax2.set_yticks([])
ax2.set_ylabel("")
ax2.set_title("Divergence at cell centers", fontsize=14)
plt.show()

# The discrete divergence operator is a sparse matrix that maps
# from faces to cell centers. To demonstrate this, we construct
# a small 2D mesh. We then show the ordering of the elements in
# the original discrete quantity :math:`\mathbf{u}` and its
# discrete divergence :math:`\boldsymbol{\phi}` as well as a
# spy plot.

mesh = TensorMesh([[(1, 6)], [(1, 3)]])
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(211)
mesh.plot_grid(ax=ax1)
ax1.plot(
    mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8
)
for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
    ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="r")
ax1.plot(
    mesh.faces_x[:, 0], mesh.faces_x[:, 1], "g>", markersize=8
)
for ii, loc in zip(range(mesh.nFx), mesh.faces_x):
    ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="g")
ax1.plot(
    mesh.faces_y[:, 0], mesh.faces_y[:, 1], "g^", markersize=8
)
for ii, loc in zip(range(mesh.nFy), mesh.faces_y):
    ax1.text(loc[0] + 0.05, loc[1] + 0.1, "{0:d}".format((ii + mesh.nFx)), color="g")

ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['bottom'].set_color('white')
ax1.spines['top'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.spines['right'].set_color('white')
ax1.set_xlabel('X', fontsize=16, labelpad=-5)
ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
ax1.set_title("Mapping of Face Divergence", fontsize=14, pad=15)
ax1.legend(
    ['Mesh', '$\mathbf{\phi}$ (centers)', '$\mathbf{u}$ (faces)'],
    loc='upper right', fontsize=14
)
ax2 = fig.add_subplot(212)
ax2.spy(mesh.face_divergence)
ax2.set_title("Spy Plot", fontsize=14, pad=5)
ax2.set_ylabel("Cell Index", fontsize=12)
ax2.set_xlabel("Face Index", fontsize=12)
plt.show()
