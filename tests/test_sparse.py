import numpy as np
import scipy.sparse
import tectosaur.util.sparse as sparse

import logging
logger = logging.getLogger(__name__)

def test_bsrmv():
    A = np.zeros((4,4))
    A[:2,:2] = np.random.rand(2,2)
    A[2:,2:] = np.random.rand(2,2)
    A_bsr = sparse.from_scipy_bsr(scipy.sparse.bsr_matrix(A))
    x = np.random.rand(A.shape[1])
    correct = A.dot(x)
    out = A_bsr.dot(x)
    np.testing.assert_almost_equal(out, correct)

def test_dense_bsrmv():
    A = np.random.rand(100,100)
    A_bsr = sparse.from_scipy_bsr(scipy.sparse.bsr_matrix(A))
    x = np.random.rand(A.shape[1])
    correct = A.dot(x)
    out = A_bsr.dot(x)
    np.testing.assert_almost_equal(out, correct)

def dense_to_coo(A, blocksize):
    assert(A.shape[0] % blocksize == 0)
    assert(A.shape[1] % blocksize == 0)
    n_block_rows = A.shape[0] // blocksize
    n_block_cols = A.shape[1] // blocksize
    data = np.swapaxes(
        A.reshape((n_block_rows,blocksize,n_block_cols,blocksize)), 1, 2
    ).reshape((-1, blocksize, blocksize)).copy()
    rows = np.tile(np.arange(n_block_rows)[:,np.newaxis], (1, n_block_cols)).flatten().copy()
    cols = np.tile(np.arange(n_block_cols)[np.newaxis, :], (n_block_rows, 1)).flatten().copy()
    return rows, cols, data

def dense_to_coo_tester(shape):
    A = np.random.rand(*shape)
    bs = 4
    rows, cols, data = dense_to_coo(A, bs)
    A_re = np.empty(A.shape)
    for i in range(rows.shape[0]):
        A_re[(bs*rows[i]):(bs*rows[i]+bs), (bs*cols[i]):(bs*cols[i]+bs)] = data[i]
    np.testing.assert_almost_equal(A, A_re)

def test_dense_to_coo():
    dense_to_coo_tester((100, 100))
    dense_to_coo_tester((60, 100))
    dense_to_coo_tester((100, 60))

def dense_bcoomv_tester(shape):
    A = np.random.rand(*shape)
    x = np.random.rand(A.shape[1])
    correct = A.dot(x)
    rows, cols, data = dense_to_coo(A, 4)
    A_bcoo = sparse.BCOOMatrix(rows, cols, data, A.shape)
    out = A_bcoo.dot(x)
    np.testing.assert_almost_equal(out, correct)

def test_dense_bcoomv():
    dense_bcoomv_tester((100, 100))
    dense_bcoomv_tester((60, 100))
    dense_bcoomv_tester((100, 60))

def test_to_bsr():
    A = np.random.rand(100,100)
    x = np.random.rand(A.shape[1])
    rows, cols, data = dense_to_coo(A, 4)
    A_bcoo = sparse.BCOOMatrix(rows, cols, data, A.shape)
    np.testing.assert_almost_equal(A_bcoo.dot(x), A.dot(x))
    A_bsr = A_bcoo.to_bsr()
    np.testing.assert_almost_equal(A_bsr.dot(x), A.dot(x))

def benchmark_bsrmv():
    from tectosaur.util.timer import Timer

    float_type = np.float32
    blocksize = 9
    nb = 100000
    nnzbs = 5000000
    t = Timer()
    rows = np.random.randint(nb, size = (nnzbs))
    cols = np.random.randint(nb, size = (nnzbs))
    data = np.ones((nnzbs, blocksize, blocksize))
    x = np.random.rand(nb * blocksize).astype(float_type)
    t.report('random')

    mat_bcoo = sparse.BCOOMatrix(rows, cols, data, (nb * blocksize, nb * blocksize))
    t.report('make bcoo')

    mat_bsr = mat_bcoo.to_bsr()
    t.report('make bsr')

    mat_sp = mat_bsr.to_scipy()
    t.report('make scipy')

    for i in range(3):
        y = mat_bsr.dot(x)
        t.report('bsr mv')

        y2 = mat_bcoo.dot(x)
        t.report('bcoo mv')

        correct = mat_sp.dot(x)
        t.report('scipy mv')

    np.testing.assert_almost_equal(y, correct, 2)
    np.testing.assert_almost_equal(y2, correct, 2)

if __name__ == "__main__":
    benchmark_bsrmv()
