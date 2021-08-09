# For the 4 classifications allowable (scalar, isotropic, anistropic and tensor),
# we construct the property tensor on a small 2D mesh. We then construct the
# inverse of the property tensor and compare.

from discretize.utils import make_property_tensor, inverse_property_tensor
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

# Construct the inverse property tensor in each case

Minv_scalar = inverse_property_tensor(mesh, sigma_scalar, return_matrix=True)
Minv_isotropic = inverse_property_tensor(mesh, sigma_isotropic, return_matrix=True)
Minv_anisotropic = inverse_property_tensor(mesh, sigma_anisotropic, return_matrix=True)
Minv_tensor = inverse_property_tensor(mesh, sigma_tensor, return_matrix=True)

# Plot the property tensors.
# Lists for plotting
M_list = [M_scalar, M_isotropic, M_anisotropic, M_tensor]
Minv_list = [Minv_scalar, Minv_isotropic, Minv_anisotropic, Minv_tensor]
case_list = ['Scalar', 'Isotropic', 'Anisotropic', 'Full Tensor']
fig1 = plt.figure(figsize=(15, 4))
ax1 = 4*[None]
for ii in range(0, 4):
    ax1[ii] = fig1.add_axes([0.05+0.22*ii, 0.05, 0.18, 0.9])
    ax1[ii].imshow(
        M_list[ii].todense(), interpolation='none', cmap='binary', vmax=10.
    )
    ax1[ii].set_title('$M$ (' + case_list[ii] + ')', fontsize=24)
cax1 = fig1.add_axes([0.92, 0.15, 0.01, 0.7])
norm1 = mpl.colors.Normalize(vmin=0., vmax=10.)
cbar1 = mpl.colorbar.ColorbarBase(
    cax1, norm=norm1, orientation="vertical", cmap=mpl.cm.binary
)
plt.show()

# Plot the inverse property tensors.

fig2 = plt.figure(figsize=(15, 4))
ax2 = 4*[None]
for ii in range(0, 4):
    ax2[ii] = fig2.add_axes([0.05+0.22*ii, 0.05, 0.18, 0.9])
    ax2[ii].imshow(
        Minv_list[ii].todense(), interpolation='none', cmap='binary', vmax=1.
    )
    ax2[ii].set_title('$M^{-1}$ (' + case_list[ii] + ')', fontsize=24)
cax2 = fig2.add_axes([0.92, 0.15, 0.01, 0.7])
norm2 = mpl.colors.Normalize(vmin=0., vmax=1.)
cbar2 = mpl.colorbar.ColorbarBase(
    cax2, norm=norm2, orientation="vertical", cmap=mpl.cm.binary
)
plt.show()
