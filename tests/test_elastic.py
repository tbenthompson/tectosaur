from tectosaur.tensors import *
from tectosaur.elastic import *
import numpy as np
import dill
import sympy
from slow_test import slow

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

    def kernel_tester(k_idx, i, j):
        kernel = kernel_builders[k_idx](i, j)
        kernel_lambda = sympy.utilities.lambdify(all_args, kernel['expr'], "numpy")

        for t_idx in range(10):
            params = np.random.rand(16, 1)
            params[-3:] /= np.linalg.norm(params[-3:])
            params[-6:-3] /= np.linalg.norm(params[-6:-3])
            totest = kernel_lambda(*params)
            exact = all_kernels[k_idx][i][j](*params)
            error = np.abs((exact - totest) / totest)
            assert(abs(error) < 1e-10)

    for k_idx in range(4):
        for i in range(3):
            for j in range(3):
                kernel_tester(k_idx, i, j)
