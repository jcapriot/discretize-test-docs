# Here, we define four 2x2 matricies and reorganize their elements into
# 4 vectors a11, a12, a21 and a22. We then examine the outputs of the
# function **inverse_2x2_block_diagonal** when the argument
# *return_matrix* is set to both *True* and *False*.

from discretize.utils import inverse_2x2_block_diagonal
import numpy as np
import matplotlib.pyplot as plt

# Define four 3x3 matricies, and organize their elements into four vectors

A1 = np.random.uniform(1, 10, (2, 2))
A2 = np.random.uniform(1, 10, (2, 2))
A3 = np.random.uniform(1, 10, (2, 2))
A4 = np.random.uniform(1, 10, (2, 2))
[[a11, a12], [a21, a22]] = np.stack([A1, A2, A3, A4], axis=-1)

# Return the elements of their inverse and validate

b11, b12, b21, b22 = inverse_2x2_block_diagonal(
    a11, a12, a21, a22, return_matrix=False
)
Bs = np.stack([[b11, b12],[b21, b22]])
B1, B2, B3, B4 = Bs.transpose((2, 0, 1))

np.linalg.inv(A1)
# array([[ 0.34507439, -0.4831833 ],
# [-0.24286626,  0.57531461]])
B1
# array([[ 0.34507439, -0.4831833 ],
# [-0.24286626,  0.57531461]])

# Plot the sparse block matrix containing elements of the inverses

M = inverse_2x2_block_diagonal(
    a11, a12, a21, a22
)
plt.spy(M)
plt.show()
