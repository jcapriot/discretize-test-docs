from discretize import tests, utils
import numpy as np
import matplotlib.pyplot as plt

def simplePass(x):
    return np.sin(x), utils.sdiag(np.cos(x))
passed = tests.checkDerivative(simplePass, np.random.randn(5))
# ==================== checkDerivative ====================
# iter    h         |ft-f0|   |ft-f0-h*J0*dx|  Order
# ---------------------------------------------------------
# 0   1.00e-01    1.690e-01     8.400e-03      nan
# 1   1.00e-02    1.636e-02     8.703e-05      1.985
# 2   1.00e-03    1.630e-03     8.732e-07      1.999
# 3   1.00e-04    1.629e-04     8.735e-09      2.000
# 4   1.00e-05    1.629e-05     8.736e-11      2.000
# 5   1.00e-06    1.629e-06     8.736e-13      2.000
# 6   1.00e-07    1.629e-07     8.822e-15      1.996
# ========================= PASS! =========================
# Once upon a time, a happy little test passed.
