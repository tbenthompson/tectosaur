import numpy as np
import tectosaur.util.gpu as gpu
from cuda_cffi import cusparse as csp
csp.init()

class cusparseBSR:
    def __init__(self, scipy_mat):
        self.shape = scipy_mat.shape
        self.blocksize = scipy_mat.blocksize[0]
        assert(scipy_mat.blocksize[1] == self.blocksize)
        self.blockshape = (self.shape[0] // self.blocksize, self.shape[1] // self.blocksize)
        self.gpu_indptr = gpu.to_gpu(scipy_mat.indptr, np.int32)
        self.gpu_indices = gpu.to_gpu(scipy_mat.indices, np.int32)
        self.gpu_data = gpu.to_gpu(scipy_mat.data, scipy_mat.dtype)
        self.descr = csp.cusparseCreateMatDescr()
        csp.cusparseSetMatType(self.descr, csp.CUSPARSE_MATRIX_TYPE_GENERAL)
        csp.cusparseSetMatIndexBase(self.descr, csp.CUSPARSE_INDEX_BASE_ZERO)
        if self.gpu_data.dtype == np.float32:
            self.mv_fn = csp.cusparseSbsrmv
        elif self.gpu_data.dtype == np.float64:
            self.mv_fn = csp.cusparseDbsrmv

    def dot(self, x):
        handle = csp.misc._global_cusparse_handle
        transA = csp.CUSPARSE_OPERATION_NON_TRANSPOSE
        dir = csp.CUSPARSE_DIRECTION_ROW
        mb = self.blockshape[0]
        nb = self.blockshape[1]
        nnzb = self.gpu_data.shape[0]
        alpha = 1.0
        beta = 0.0

        gpu_x = gpu.to_gpu(x, self.gpu_data.dtype)
        gpu_y = gpu.empty_gpu(self.shape[0], self.gpu_data.dtype)

        # cusparseSbsrmv(cusparseHandle_t handle, cusparseDirection_t dir,
        #     cusparseOperation_t trans, int mb, int nb, int nnzb,
        #     const float *alpha, const cusparseMatDescr_t descr,
        #     const float *bsrVal, const int *bsrRowPtr, const int *bsrColInd,
        #     int  blockDim, const float *x,
        #     const float *beta, float *y)
        self.mv_fn(
            handle, transA, dir, mb, nb, nnzb, alpha,
            self.descr, self.gpu_data, self.gpu_indptr,
            self.gpu_indices, self.blocksize, gpu_x, beta, gpu_y
        )
        return gpu_y.get()
