import numpy as np

from tectosaur.kernels.tensors import *
from tectosaur.util.test_decorators import slow

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
    import sympy
    from tectosaur.farfield import farfield_pts_direct
    import dill
    from tectosaur.kernels.elastic import U,T,A,H,all_args
    with open('tests/3d_kernels.pkl', 'rb') as f:
        all_kernels = dill.load(f)

    kernel_builders = [U, T, A, H]
    name_from_idx = ['elasticU3', 'elasticT3', 'elasticA3', 'elasticH3']

## 0.999912664648 [ 1.  0.  0.] [[ 0.04484328  0.61542118  0.61866593]] [[ 0.42195645  0.26927334  0.34952787]] 0.578332
## 0.999884855908 [ 1.  0.  0.] [[ 0.64798468  0.75984156  0.9303776 ]] [[ 0.9142682  0.6666224  0.2419737]] 0.743974
## 0.353440002049 0.230722118684 [ 0.  1.  0.] [[ 0.51952446  0.50394344  0.58011067]] [[ 0.30859265  0.47683924  0.82304007]] [[ 0.11135003  0.75749063  0.58626622]] [[ 0.15490565  0.72846574  0.66733944]] 0.480552
## 0.313304293459 0.745684163724 [ 0.  0.  1.] [[ 0.78027838  0.45547488  0.23808946]] [[ 0.52706057  0.46850777  0.70901877]] [[ 0.78663123  0.44327369  0.29162556]] [[ 0.13285404  0.71820003  0.68303621]] 0.0552751

    def call_gpu_kernel(k_idx, i, j, params):
        V = np.zeros(3, dtype = np.float32)
        V[j] = 1.0
        src_pt = np.array([params[4:7]])
        obs_pt = np.array([params[7:10]])
        src_n = np.array([params[10:13]])
        obs_n = np.array([params[13:16]])
        nbody_result = farfield_pts_direct(
            name_from_idx[k_idx], obs_pt, obs_n,
            src_pt, src_n, V, params[0:2],
            np.float64
        ).reshape((1,3))
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

            totest_gpu = call_gpu_kernel(k_idx, i, j, params)
            error = np.abs((exact - totest_gpu) / exact)
            assert(error < 1e-4)

    for k_idx in range(4):
        for i in range(3):
            for j in range(3):
                kernel_tester(k_idx, i, j)
