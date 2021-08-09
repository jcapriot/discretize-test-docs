# Below, we demonstrate how to set boundary conditions for the
# x-component cell gradient stencil, construct the operator and apply it
# to a discrete scalar quantity. The mapping of the operator and
# its sparsity is also illustrated. Our example is carried out on a 2D
# mesh but it can be done equivalently for a 3D mesh.

# We start by importing the necessary packages and modules.

from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# We then construct a mesh and define a scalar function at cell
# centers. In this case, the scalar represents some block within
# a homogeneous medium.

# Create a uniform grid

h = np.ones(40)
mesh = TensorMesh([h, h], "CC")

# Create a discrete scalar at cell centers

centers = mesh.cell_centers
phi = np.zeros(mesh.nC)
k = (np.abs(mesh.cell_centers[:, 0]) < 10.) & (np.abs(mesh.cell_centers[:, 1]) < 10.)
phi[k] = 1.

# Before constructing the stencil gradient operator, we must define
# the boundary conditions; zero Neumann for our example. Even though
# we are only computing the difference along x, we define boundary
# conditions for all boundary faces. Once the
# operator is created, it is applied as a matrix-vector product.

mesh.set_cell_gradient_BC(['neumann', 'neumann'])
Gx = mesh.stencil_cell_gradient_x
diff_phi_x = Gx @ phi

# Now we plot the original scalar, and the differencing taken along the
# x axes.

fig = plt.figure(figsize=(13, 5))
ax1 = fig.add_subplot(121)
mesh.plot_image(phi, ax=ax1)
ax1.set_title("Scalar at cell centers", fontsize=14)

ax2 = fig.add_subplot(122)
v = np.r_[diff_phi_x, np.zeros(mesh.nFy)]  # Define vector for plotting fun
mesh.plot_image(v, ax=ax2, v_type="Fx")
ax2.set_yticks([])
ax2.set_ylabel("")
ax2.set_title("Difference (x-axis)", fontsize=14)
plt.show()

# The x-component cell gradient stencil is a sparse differencing matrix
# that maps from cell centers to x-faces. To demonstrate this, we construct
# a small 2D mesh. We then show the ordering of the elements
# and a spy plot.

mesh = TensorMesh([[(1, 3)], [(1, 4)]])
mesh.set_cell_gradient_BC('neumann')

fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121)
mesh.plot_grid(ax=ax1)
ax1.set_title("Mapping of Stencil", fontsize=14, pad=15)

ax1.plot(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8)
for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
    ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")
ax1.plot(mesh.faces_x[:, 0], mesh.faces_x[:, 1], "g>", markersize=8)
for ii, loc in zip(range(mesh.nFx), mesh.faces_x):
    ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="g")
ax1.set_xticks([])
ax1.set_yticks([])
ax1.spines['bottom'].set_color('white')
ax1.spines['top'].set_color('white')
ax1.spines['left'].set_color('white')
ax1.spines['right'].set_color('white')
ax1.set_xlabel('X', fontsize=16, labelpad=-5)
ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
ax1.legend(
    ['Mesh', '$\mathbf{\phi}$ (centers)', '$\mathbf{Gx^* \phi}$ (x-faces)'],
    loc='upper right', fontsize=14
)
ax2 = fig.add_subplot(122)
ax2.spy(mesh.stencil_cell_gradient_x)
ax2.set_title("Spy Plot", fontsize=14, pad=5)
ax2.set_ylabel("X-Face Index", fontsize=12)
ax2.set_xlabel("Cell Index", fontsize=12)
plt.show()
