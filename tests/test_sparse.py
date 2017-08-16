import numpy as np
import scipy.sparse
import tectosaur.util.sparse as sparse

from tectosaur.util.logging import setup_logger
logger = setup_logger(__name__)

def test_bsrmv():
    A = np.zeros((4,4))
    A[:2,:2] = np.random.rand(2,2)
    A[2:,2:] = np.random.rand(2,2)
    A_bsr = scipy.sparse.bsr_matrix(A)
    x = np.random.rand(A.shape[1])
    correct = A.dot(x)
    out = sparse.bsr_mv(A_bsr, x)
    np.testing.assert_almost_equal(out, correct)

def test_dense_bsrmv():
    A = np.random.rand(100,100)
    A_bsr = scipy.sparse.bsr_matrix(A)
    x = np.random.rand(A.shape[1])
    correct = A.dot(x)
    out = sparse.bsr_mv(A_bsr, x)
    np.testing.assert_almost_equal(out, correct)

def benchmark_bsrmv():
    from tectosaur.util.timer import Timer
    from cppimport import cppimport
    fast_assembly = cppimport("tectosaur.ops.fast_assembly")

    float_type = np.float32
    int_type = np.int32
    blocksize = 9
    nb = 100000
    nnzbs = 5000000
    t = Timer()
    rows = np.random.randint(nb, size = (nnzbs))
    cols = np.random.randint(nb, size = (nnzbs))
    data = np.ones((nnzbs, blocksize, blocksize))
    x = np.random.rand(nb * blocksize).astype(float_type)
    t.report('random')

    data, indices, indptr = fast_assembly.make_bsr_matrix(
        nb * blocksize, nb * blocksize, blocksize, blocksize, data, rows, cols
    )
    mat = scipy.sparse.bsr_matrix((data.astype(float_type), indices.astype(int_type), indptr.astype(int_type)))
    t.report('make')
    for i in range(3):
        y = sparse.bsr_mv(mat, x)
        t.report('bsr mv')
        correct = mat.dot(x)
        t.report('scipy mv')
    np.testing.assert_almost_equal(y, correct, 2)

if __name__ == "__main__":
    benchmark_bsrmv()
