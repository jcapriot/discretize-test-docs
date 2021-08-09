# Here, we define four 3x3 matricies and reorganize their elements into
# 9 vectors a11, a12, ..., a33. We then examine the outputs of the
# function **inverse_3x3_block_diagonal** when the argument
# *return_matrix* is set to both *True* and *False*.

from discretize.utils import inverse_3x3_block_diagonal
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Define four 3x3 matricies, and organize their elements into nine vectors

A1 = np.random.uniform(1, 10, (3, 3))
A2 = np.random.uniform(1, 10, (3, 3))
A3 = np.random.uniform(1, 10, (3, 3))
A4 = np.random.uniform(1, 10, (3, 3))
[[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]] = np.stack(
    [A1, A2, A3, A4], axis=-1
)

# Return the elements of their inverse and validate

b11, b12, b13, b21, b22, b23, b31, b32, b33 = inverse_3x3_block_diagonal(
    a11, a12, a13, a21, a22, a23, a31, a32, a33, return_matrix=False
)
Bs = np.stack([[b11, b12, b13],[b21, b22, b23],[b31, b32, b33]])
B1, B2, B3, B4 = Bs.transpose((2, 0, 1))

np.linalg.inv(A1)
# array([[ 0.20941584,  0.18477151, -0.22637147],
# [-0.06420656, -0.34949639,  0.29216461],
# [-0.14226339,  0.11160555,  0.0907583 ]])
B1
# array([[ 0.20941584,  0.18477151, -0.22637147],
# [-0.06420656, -0.34949639,  0.29216461],
# [-0.14226339,  0.11160555,  0.0907583 ]])

# We can also return this as a sparse matrix with block diagonal inverse

M = inverse_3x3_block_diagonal(
    a11, a12, a13, a21, a22, a23, a31, a32, a33
)
plt.spy(M)
plt.show()
