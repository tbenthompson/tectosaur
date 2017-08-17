import scipy.sparse
import numpy as np

import cppimport
fast_sparse = cppimport.cppimport("tectosaur.util.fast_sparse")

def get_mv_fnc(base_name, dtype, blocksize):
    fnc_name = base_name + str(blocksize)
    if dtype == np.float32:
        return getattr(fast_sparse, 's' + fnc_name)
    else:
        return getattr(fast_sparse, 'd' + fnc_name)

class BCOOMatrix:
    def __init__(self, rows, cols, data, shape):
        self.rows = rows
        self.cols = cols
        self.data = data
        self.shape = shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def blocksize(self):
        return self.data.shape[1]

    def dot(self, v):
        out = np.zeros(self.shape[1], dtype = self.dtype)
        fnc = get_mv_fnc('bcoomv', self.dtype, self.blocksize)
        fnc(self.rows, self.cols, self.data, v.astype(self.dtype), out)
        return out

    def to_bsr(self):
        indptr, indices, data = fast_sparse.make_bsr_matrix(
            *self.shape, self.data, self.rows, self.cols
        )
        return BSRMatrix(indptr, indices, data, self.shape)

    def to_dense(self):
        return self.to_bsr().to_scipy().to_dense()

class BSRMatrix:
    def __init__(self, indptr, indices, data, shape):
        assert(data.shape[1] == data.shape[2])
        self.indptr = indptr
        self.indices = indices
        self.data = data
        self.shape = shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def blocksize(self):
        return self.data.shape[1]

    def dot(self, v):
        out = np.empty(self.shape[1], dtype = self.dtype)
        fnc = get_mv_fnc('bsrmv', self.dtype, self.blocksize)
        fnc(self.indptr, self.indices, self.data, v.astype(self.dtype), out)
        return out

    def to_scipy(self):
        return scipy.sparse.bsr_matrix((self.data, self.indices, self.indptr), self.shape)

def from_scipy_bsr(A):
    return BSRMatrix(A.indptr, A.indices, A.data, A.shape)
