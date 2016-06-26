import pycuda.autoinit
import numpy as np
import skcuda.linalg as culg

from tectosaur.linalg import gpu_mvp, block_gpu_mvp
from tectosaur.timer import Timer
import tectosaur.gpu

culg.init()
def test_gpu_mvp():
    A = np.random.rand(100, 100).astype(np.float32)
    x = np.random.rand(100, 1).astype(np.float32)
    res = gpu_mvp(A, x)
    np.testing.assert_almost_equal(res, A.dot(x), 5)
