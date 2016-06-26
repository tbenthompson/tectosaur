import numpy as np
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culg

def gpu_mvp(A, x):
    assert(A.dtype == np.float32)
    assert(x.dtype == np.float32)
    if type(A) != gpuarray.GPUArray:
        A = gpuarray.to_gpu(A)
    x_gpu = gpuarray.to_gpu(x)
    Ax_gpu = culg.dot(A, x_gpu)
    return Ax_gpu.get()
