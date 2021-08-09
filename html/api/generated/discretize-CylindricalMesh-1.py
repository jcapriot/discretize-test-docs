# To create a general 3D cylindrical mesh, we discretize along the radial,
# azimuthal and vertical axis. For example:

from discretize import CylindricalMesh
import matplotlib.pyplot as plt
import numpy as np

ncr = 10  # number of mesh cells in r
ncp = 8   # number of mesh cells in phi
ncz = 15  # number of mesh cells in z
dr = 15   # cell width r
dz = 10   # cell width z

hr = dr * np.ones(ncr)
hp = (2 * np.pi / ncp) * np.ones(ncp)
hz = dz * np.ones(ncz)
mesh1 = CylindricalMesh([hr, hp, hz])

mesh1.plot_grid()
plt.show()

# For a cylindrically symmetric mesh, the disretization along the
# azimuthal direction is set with a flag of *1*. This reduces the
# size of numerical systems given that the derivative along the
# azimuthal direction is 0. For example:

ncr = 10      # number of mesh cells in r
ncz = 15      # number of mesh cells in z
dr = 15       # cell width r
dz = 10       # cell width z
npad_r = 4    # number of padding cells in r
npad_z = 4    # number of padding cells in z
exp_r = 1.25  # expansion rate of padding cells in r
exp_z = 1.25  # expansion rate of padding cells in z

# A value of 1 is used to define the discretization in phi for this case.

hr = [(dr, ncr), (dr, npad_r, exp_r)]
hz = [(dz, npad_z, -exp_z), (dz, ncz), (dz, npad_z, exp_z)]
mesh2 = CylindricalMesh([hr, 1, hz])

mesh2.plot_grid()
plt.show()
