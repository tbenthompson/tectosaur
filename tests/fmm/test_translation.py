import numpy as np
import tectosaur.fmm.fmm_wrapper as fmm
from tectosaur.util.test_decorators import golden_master
from tectosaur.fmm.c2e import *

def test_inscribe():
    b = fmm.three.Ball([1,1,1], 2.0)
    s = inscribe_surf(b, 0.5, np.array([[1,0,0],[0,1,0],[0,0,1]]))
    np.testing.assert_almost_equal(s, [[2,1,1],[1,2,1],[1,1,2]])

# @golden_master(5)
# def test_c2e_solve(request):
def test_c2e_solve():
    b = fmm.three.Ball([0, 0, 0], 1.3)
    s = surrounding_surface(100, 3)
    K_name = 'elasticH3'
    K = fmm.kernels[K_name]
    params = np.array([1.0, 0.25])
    float_type = np.float64
    gpu_module = fmm.get_gpu_module(s, K, float_type)
    op = c2e_solve(gpu_module, s, b, 3.0, 1.1, K, params, float_type)

    cfg = fmm.three.FMMConfig(3.0, 1.1, 100, K_name, params)
    op2 = np.array(fmm.three.c2e_solve(s.tolist(), b, 3.0, 1.1, cfg))
    np.testing.assert_almost_equal(op, op2)
    # return op

def test_scaling_relations():
    b = fmm.two.Ball([0,0], 1.0)
    s = surrounding_surface(5, 2)
    cfg = fmm.two.FMMConfig(3.0, 1.1, 5, "laplaceD2", [])
    op = np.array(fmm.two.c2e_solve(s.tolist(), b, 3.0, 1.1, cfg))

    for i in range(5):
        b2 = fmm.two.Ball(np.random.rand(2).tolist(), 1.0)
        op2 = np.array(fmm.two.c2e_solve(s.tolist(), b2, 3.0, 1.1, cfg))
        np.testing.assert_almost_equal(op2, op)

    scale = 2.0
    b2 = fmm.two.Ball([0,0], scale)
    op2 = np.array(fmm.two.c2e_solve(s.tolist(), b2, 3.0, 1.1, cfg)) / scale
    np.testing.assert_almost_equal(op2, op)
