import numpy as np
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culg

def block_gpu_mvp(A, x, block_size):
    assert(A.shape[1] == x.shape[0])
    result = np.zeros((A.shape[0], 1))
    row_start = 0
    while row_start < A.shape[0]:
        row_end = min(row_start + block_size,A.shape[0])
        col_start = 0
        while col_start < A.shape[1]:
            col_end = min(col_start + block_size,A.shape[1])
            A_gpu = gpuarray.to_gpu(A[row_start:row_end, col_start:col_end])
            x_gpu = gpuarray.to_gpu(x[col_start:col_end])
            Ax_gpu = culg.dot(A_gpu, x_gpu)
            result[row_start:row_end,:] += Ax_gpu.get()
            col_start = col_end
        row_start = row_end
    return result

def gpu_mvp(A, x):
    assert(A.dtype == np.float32)
    assert(x.dtype == np.float32)
    if type(A) != gpuarray.GPUArray:
        A = gpuarray.to_gpu(A)
    x_gpu = gpuarray.to_gpu(x)
    Ax_gpu = culg.dot(A, x_gpu)
    return Ax_gpu.get()
