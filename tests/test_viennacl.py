import numpy as np

import tectosaur.viennacl

def test_prod():
    n = 10
    A = np.random.rand(n,n).astype(np.float32)
    x = np.random.rand(n).astype(np.float32)
    correct = A.dot(x)

    tectosaur.viennacl.prod(A, x, np.float32)

