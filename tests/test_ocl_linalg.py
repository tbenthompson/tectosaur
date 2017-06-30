import time
import numpy as np

from tectosaur.util.test_decorators import slow

import tectosaur.viennacl as viennacl
float_type = np.float32

@slow
def test_mat_vec():
    n = 32 * 32 * 1
    A = np.random.rand(n,n).astype(float_type)
    x = np.random.rand(n).astype(float_type)
    correct = A.dot(x)
    vals = viennacl.prod(A, x, float_type)
    assert(np.max(np.abs((correct - vals) / correct)) < n * 1e-05)
