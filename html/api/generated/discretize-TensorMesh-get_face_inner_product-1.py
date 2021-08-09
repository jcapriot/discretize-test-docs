# Here we provide some examples of face inner product matrices.
# For simplicity, we will work on a 2 x 2 x 2 tensor mesh.
# As seen below, we begin by constructing and imaging the basic
# face inner product matrix.

from discretize import TensorMesh
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

h = np.ones(2)
mesh = TensorMesh([h, h, h])
Mf = mesh.get_face_inner_product()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.imshow(Mf.todense())
ax.set_title('Basic Face Inner Product Matrix', fontsize=18)
plt.show()

# Next, we consider the case where the physical properties
# of the cells are defined by consistutive relations. For
# the isotropic, diagonal anisotropic and full tensor cases,
# we show the physical property tensor for a single cell.

# Define 4 constitutive parameters and define the tensor
# for each cell for isotropic, diagonal and tensor cases.

sig1, sig2, sig3, sig4, sig5, sig6 = 6, 5, 4, 3, 2, 1
sig_iso_tensor = sig1 * np.eye(3)
sig_diag_tensor = np.diag(np.array([sig1, sig2, sig3]))
sig_full_tensor = np.array([
    [sig1, sig4, sig5],
    [sig4, sig2, sig6],
    [sig5, sig6, sig3]
])

# Then plot matrix entries,

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131)
ax1.imshow(sig_iso_tensor)
ax1.axis('off')
ax1.set_title("Tensor (isotropic)", fontsize=16)
ax2 = fig.add_subplot(132)
ax2.imshow(sig_diag_tensor)
ax2.axis('off')
ax2.set_title("Tensor (diagonal anisotropic)", fontsize=16)
ax3 = fig.add_subplot(133)
ax3.imshow(sig_full_tensor)
ax3.axis('off')
ax3.set_title("Tensor (full anisotropic)", fontsize=16)
plt.show()

# Here, construct and image the face inner product matrices for
# the isotropic, diagonal anisotropic and full tensor cases.
# Spy plots are used to demonstrate the sparsity of the inner
# product matrices.

# Isotropic case:

v = np.ones(mesh.nC)
sig = sig1 * v
M1 = mesh.get_face_inner_product(sig)

# Diagonal anisotropic case:

sig = np.c_[sig1*v, sig2*v, sig3*v]
M2 = mesh.get_face_inner_product(sig)

# Full anisotropic case:

sig = np.tile(np.c_[sig1, sig2, sig3, sig4, sig5, sig6], (mesh.nC, 1))
M3 = mesh.get_face_inner_product(sig)

# And then we can plot the sparse representation,

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(131)
ax1.spy(M1, ms=5)
ax1.set_title("M (isotropic)", fontsize=16)
ax2 = fig.add_subplot(132)
ax2.spy(M2, ms=5)
ax2.set_title("M (diagonal anisotropic)", fontsize=16)
ax3 = fig.add_subplot(133)
ax3.spy(M3, ms=5)
ax3.set_title("M (full anisotropic)", fontsize=16)
plt.show()
