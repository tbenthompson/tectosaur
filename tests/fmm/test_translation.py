import numpy as np
import tectosaur.fmm.fmm as fmm
from tectosaur.util.test_decorators import golden_master
from tectosaur.fmm.surrounding_surf import surrounding_surf
from tectosaur.fmm.c2e import inscribe_surf, c2e_solve, Ball

float_type = np.float64

def test_inscribe():
    b = Ball([1,1,1], 2.0)
    s = inscribe_surf(b, 0.5, np.array([[1,0,0],[0,1,0],[0,0,1]]))
    np.testing.assert_almost_equal(s, [[2,1,1],[1,2,1],[1,1,2]])

@golden_master(5)
def test_c2e_solve(request):
    b = Ball([0, 0, 0], 1.3)
    s = surrounding_surf(50, 3)
    K_name = 'elasticH3'
    params = np.array([1.0, 0.25])
    cfg = fmm.make_config(K_name, params, 1.0, 1.0, 1, float_type)
    op = c2e_solve(cfg.gpu_module, s, b, 3.0, 1.1, cfg.K, params, float_type)
    return op

def test_scaling_relations():
    b = Ball([0,0], 1.0)
    s = surrounding_surf(5, 2)

    K_name = 'laplaceD2'
    cfg = fmm.make_config(K_name, [], 1.0, 1.0, 1, float_type)
    op = np.array(c2e_solve(cfg.gpu_module, s, b, 3.0, 1.1, cfg.K, cfg.params, float_type))

    for i in range(5):
        b2 = Ball(np.random.rand(2).tolist(), 1.0)
        op2 = np.array(c2e_solve(cfg.gpu_module, s, b2, 3.0, 1.1, cfg.K, cfg.params, float_type))
        np.testing.assert_almost_equal(op2, op)

    scale = 2.0
    b2 = Ball([0,0], scale)
    op2 = np.array(c2e_solve(cfg.gpu_module, s, b2, 3.0, 1.1, cfg.K, cfg.params, float_type)) / scale
    np.testing.assert_almost_equal(op2, op)
