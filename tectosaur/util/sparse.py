import numpy as np
import cppimport

fast_sparse = cppimport.cppimport("tectosaur.util.fast_sparse")

def bsr_mv(A, x):
    out = np.empty(A.shape[1], dtype = A.dtype)
    fnc_name = 'bsrmv' + str(A.blocksize[0])
    assert(A.blocksize[0] == A.blocksize[1])
    if A.dtype == np.float32:
        fnc = getattr(fast_sparse, 's' + fnc_name)
    else:
        fnc = getattr(fast_sparse, 'd' + fnc_name)
    fnc(A.shape[0] // A.blocksize[0], A.indptr, A.indices, A.data, x, out)
    return out
