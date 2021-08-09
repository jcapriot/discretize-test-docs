# Here, we construct a 4 cell by 4 cell tensor mesh. For our first example we
# consider isotropic physical properties; that is, the physical properties
# of each cell are defined a scalar value. We construct the face inner product
# matrix and visualize it with a spy plot. We then use
# **get_face_inner_product_deriv** to construct the function handle
# :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation
# of this function on a spy plot.

from discretize import TensorMesh
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 14})
np.random.seed(45)
mesh = TensorMesh([[(1, 4)], [(1, 4)]])

# Define a model, and a random vector to multiply the derivative with,
# then we grab the respective derivative function and calculate the
# sparse matrix,

m = np.random.rand(mesh.nC)  # physical property parameters
u = np.random.rand(mesh.nF)  # vector of shape (n_faces)
Mf = mesh.get_face_inner_product(m)
F = mesh.get_face_inner_product_deriv(m)  # Function handle
dFdm_u = F(u)

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.85])
ax1.spy(Mf, ms=6)
ax1.set_title("Face Inner Product Matrix (Isotropic)", fontsize=14, pad=5)
ax1.set_xlabel("Face Index", fontsize=12)
ax1.set_ylabel("Face Index", fontsize=12)
ax2 = fig.add_axes([0.43, 0.05, 0.17, 0.8])
ax2.spy(dFdm_u, ms=6)
ax2.set_title(
    "$u^T \, \dfrac{\partial M(m)}{\partial m}$ (Isotropic)",
    fontsize=14, pad=5
)
ax2.set_xlabel("Parameter Index", fontsize=12)
ax2.set_ylabel("Face Index", fontsize=12)
plt.show()

# For our second example, the physical properties on the mesh are fully
# anisotropic; that is, the physical properties of each cell are defined
# by a tensor with parameters :math:`\sigma_1`, :math:`\sigma_2` and :math:`\sigma_3`.
# Once again we construct the face inner product matrix and visualize it with a
# spy plot. We then use **get_face_inner_product_deriv** to construct the
# function handle :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation
# of this function on a spy plot.

m = np.random.rand(mesh.nC, 3)  # anisotropic physical property parameters
u = np.random.rand(mesh.nF)     # vector of shape (n_faces)
Mf = mesh.get_face_inner_product(m)
F = mesh.get_face_inner_product_deriv(m)  # Function handle
dFdm_u = F(u)

# Plot the anisotropic derivative matrix,

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.8])
ax1.spy(Mf, ms=6)
ax1.set_title("Face Inner Product (Full Tensor)", fontsize=14, pad=5)
ax1.set_xlabel("Face Index", fontsize=12)
ax1.set_ylabel("Face Index", fontsize=12)
ax2 = fig.add_axes([0.4, 0.05, 0.45, 0.85])
ax2.spy(dFdm_u, ms=6)
ax2.set_title(
    "$u^T \, \dfrac{\partial M(m)}{\partial m} \;$ (Full Tensor)",
    fontsize=14, pad=5
)
ax2.set_xlabel("Parameter Index", fontsize=12)
ax2.set_ylabel("Face Index", fontsize=12)
plt.show()
