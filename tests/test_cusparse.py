import numpy as np
import scipy.sparse
from tectosaur.util.cusparse_bsr import cusparseBSR

def create():
    A = np.zeros((4,4))
    A[:2,:2] = np.random.rand(2,2)
    A[2:,2:] = np.random.rand(2,2)
    A_bsr = scipy.sparse.bsr_matrix(A)
    gpu_bsr = cusparseBSR(A_bsr)
    return A, gpu_bsr

def test_create():
    create()

def test_dot():
    A, gpu_bsr = create()
    x = np.random.rand(A.shape[1])
    correct = A.dot(x)
    out = gpu_bsr.dot(x)
    np.testing.assert_almost_equal(out, correct)
