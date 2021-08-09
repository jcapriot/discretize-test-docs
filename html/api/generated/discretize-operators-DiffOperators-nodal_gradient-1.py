# Below, we demonstrate how to apply the nodal gradient operator to
# a discrete scalar quantity, the mapping of the nodal gradient operator and
# its sparsity. Our example is carried out on a 2D mesh but it can
# be done equivalently for a 3D mesh.

# We start by importing the necessary packages and modules.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# For a discrete scalar quantity defined on the nodes, we take the
# gradient by constructing the gradient operator and multiplying
# as a matrix-vector product.

# Create a uniform grid

h = np.ones(20)
mesh = TensorMesh([h, h], "CC")

# Create a discrete scalar on nodes

nodes = mesh.nodes
phi = np.exp(-(nodes[:, 0] ** 2 + nodes[:, 1] ** 2) / 4 ** 2)

# Construct the gradient operator and apply to vector

Gn = mesh.nodal_gradient
grad_phi = Gn @ phi

fig = plt.figure(figsize=(13, 6))
ax1 = fig.add_subplot(121)
mesh.plot_image(phi, v_type="N", ax=ax1)
ax1.set_title("Scalar at nodes", fontsize=14)
ax2 = fig.add_subplot(122)
mesh.plot_image(
    grad_phi, ax=ax2, v_type="E", view="vec",
    stream_opts={"color": "w", "density": 1.0}
)
ax2.set_yticks([])
ax2.set_ylabel("")
ax2.set_title("Gradient at edges", fontsize=14)
plt.show()

# The nodal gradient operator is a sparse matrix that maps
# from nodes to edges. To demonstrate this, we construct
# a small 2D mesh. We then show the ordering of the elements in
# the original discrete quantity :math:`\boldsymbol{\phi}` and its
# discrete gradient as well as a spy plot.

mesh = TensorMesh([[(1, 3)], [(1, 6)]])
fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(121)
mesh.plot_grid(ax=ax1)
ax1.set_title("Mapping of Gradient Operator", fontsize=14, pad=15)
ax1.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], "ro", markersize=8)
for ii, loc in zip(range(mesh.nN), mesh.nodes):
    ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")

ax1.plot(mesh.edges_x[:, 0], mesh.edges_x[:, 1], "g>", markersize=8)
for ii, loc in zip(range(mesh.nEx), mesh.edges_x):
    ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="g")

ax1.plot(mesh.edges_y[:, 0], mesh.edges_y[:, 1], "g^", markersize=8)
for ii, loc in zip(range(mesh.nEy), mesh.edges_y):
    ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format((ii + mesh.nEx)), color="g")

ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['bottom'].set_color('white')
ax1.spines['top'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.spines['right'].set_color('white')
ax1.set_xlabel('X', fontsize=16, labelpad=-5)
ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
ax1.legend(
    ['Mesh', '$\mathbf{\phi}$ (nodes)', '$\mathbf{u}$ (edges)'],
    loc='upper right', fontsize=14
)
ax2 = fig.add_subplot(122)
ax2.spy(mesh.nodal_gradient)
ax2.set_title("Spy Plot", fontsize=14, pad=5)
ax2.set_ylabel("Edge Index", fontsize=12)
ax2.set_xlabel("Node Index", fontsize=12)
plt.plot()
