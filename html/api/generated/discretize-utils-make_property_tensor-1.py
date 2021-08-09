# For the 4 classifications allowable (scalar, isotropic, anistropic and tensor),
# we construct and compare the property tensor on a small 2D mesh. For this
# example, note the following:

# - The dimensions for all property tensors are the same
# - All property tensors, except in the case of full anisotropy are diagonal
# sparse matrices
# - For the scalar case, the non-zero elements are equal
# - For the isotropic case, the non-zero elements repreat in order for the x, y
# (and z) components
# - For the anisotropic case (diagonal anisotropy), the non-zero elements do not
# repeat
# - For the tensor caes (full anisotropy), there are off-diagonal components

from discretize.utils import make_property_tensor
from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Define a 2D tensor mesh

h = [1., 1., 1.]
mesh = TensorMesh([h, h], origin='00')

# Define a physical property for all cases (2D)

sigma_scalar = 4.
sigma_isotropic = np.random.randint(1, 10, mesh.nC)
sigma_anisotropic = np.random.randint(1, 10, (mesh.nC, 2))
sigma_tensor = np.random.randint(1, 10, (mesh.nC, 3))

# Construct the property tensor in each case

M_scalar = make_property_tensor(mesh, sigma_scalar)
M_isotropic = make_property_tensor(mesh, sigma_isotropic)
M_anisotropic = make_property_tensor(mesh, sigma_anisotropic)
M_tensor = make_property_tensor(mesh, sigma_tensor)

# Plot the property tensors.

M_list = [M_scalar, M_isotropic, M_anisotropic, M_tensor]
case_list = ['Scalar', 'Isotropic', 'Anisotropic', 'Full Tensor']
ax1 = 4*[None]
fig = plt.figure(figsize=(15, 4))
for ii in range(0, 4):
    ax1[ii] = fig.add_axes([0.05+0.22*ii, 0.05, 0.18, 0.9])
    ax1[ii].imshow(
        M_list[ii].todense(), interpolation='none', cmap='binary', vmax=10.
    )
    ax1[ii].set_title(case_list[ii], fontsize=24)
ax2 = fig.add_axes([0.92, 0.15, 0.01, 0.7])
norm = mpl.colors.Normalize(vmin=0., vmax=10.)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.binary
)
plt.show()
