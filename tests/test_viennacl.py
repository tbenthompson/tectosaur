import numpy as np

import tectosaur.viennacl.viennacl as vcl

def prod_tester(A_size, x_size):
    A = np.random.rand(*A_size).astype(np.float32)
    x = np.random.rand(*x_size).astype(np.float32)
    correct = A.dot(x)
    out = vcl.prod(A, x, np.float32).get()
    n_cols = A.shape[1]
    np.testing.assert_almost_equal(correct / n_cols, out / n_cols, 6)

def test_sgemv():
    n = 20
    prod_tester((n,n), (n,))

def test_sgemm():
    n = 20
    prod_tester((n,n), (n,n))

if __name__ == "__main__":
    test_sgemm()
