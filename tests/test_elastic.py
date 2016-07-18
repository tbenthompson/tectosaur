import numpy as np
import pycuda.driver as drv
import dill
import sympy

from tectosaur.tensors import *
from tectosaur.elastic import *
from tectosaur.nearfield_op import get_gpu_config, get_gpu_module

from test_decorators import slow

def test_tensors():
    t = [[0, -1], [1, 0]]
    t2 = transpose(tensor_sum(tensor_mult(t, 2), tensor_negate(t)))
    np.testing.assert_almost_equal(t2, [[0, 1], [-1, 0]])

def test_sym_skw():
    t = [[3, 2], [1, 0]]
    sym_t = SYM(t)
    skw_t = SKW(t)
    np.testing.assert_almost_equal(sym_t, [[3, 1.5], [1.5, 0]])
    np.testing.assert_almost_equal(skw_t, [[0, 0.5], [-0.5, 0]])

def test_outer():
    np.testing.assert_almost_equal(tensor_outer([0, 1], [1, 2]), [[0, 0], [1, 2]])

@slow
def test_kernels():
    with open('tests/3d_kernels.pkl', 'rb') as f:
        all_kernels = dill.load(f)

    kernel_builders = [U, T, A, H]
    gpu_module = get_gpu_module()
    gpu_fncs = [
        gpu_module.get_function("farfield_pts" + K)
        for K in ['U', 'T', 'A', 'H']
    ]


    def call_gpu_kernel(k_idx, i, j, params):
        V = np.zeros(3, dtype = np.float32)
        V[j] = 1.0
        nbody_result = np.empty((1, 3), dtype = np.float32)
        src_pt = np.array([params[4:7]], dtype = np.float32)
        obs_pt = np.array([params[7:10]], dtype = np.float32)
        src_n = np.array([params[10:13]], dtype = np.float32)
        obs_n = np.array([params[13:16]], dtype = np.float32)
        block = (get_gpu_config()['block_size'], 1, 1)
        grid = (int(np.ceil(1 / block[0])), 1)
        gpu_fncs[k_idx](
            drv.Out(nbody_result),
            drv.In(obs_pt), drv.In(obs_n),
            drv.In(src_pt), drv.In(src_n),
            drv.In(V),
            np.float32(params[0]), np.float32(params[1]),
            np.int32(1), np.int32(1),
            block = block,
            grid = grid
        )
        return nbody_result[0, i]

    def kernel_tester(k_idx, i, j):
        kernel = kernel_builders[k_idx](i, j)
        kernel_lambda = sympy.utilities.lambdify(all_args, kernel['expr'], "numpy")

        for t_idx in range(10):
            params = np.random.rand(16)
            params[-3:] /= np.linalg.norm(params[-3:])
            params[-6:-3] /= np.linalg.norm(params[-6:-3])
            totest = kernel_lambda(*params)
            exact = all_kernels[k_idx][i][j](*params)
            error = np.abs((exact - totest) / totest)
            assert(error < 1e-6)

            totest = call_gpu_kernel(k_idx, i, j, params)
            error = np.abs((exact - totest) / totest)
            assert(error < 1e-4)

    for k_idx in range(4):
        for i in range(3):
            for j in range(3):
                kernel_tester(k_idx, i, j)
