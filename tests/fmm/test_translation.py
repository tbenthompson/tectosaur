import numpy as np
import tectosaur.fmm.fmm_wrapper as fmm

def test_inscribe():
    b = fmm.three.Cube([1,1,1], 2.0 / np.sqrt(3))
    s = fmm.three.inscribe_surf(b, 0.5, [[1,0,0],[0,1,0],[0,0,1]])
    np.testing.assert_almost_equal(s, [[2,1,1],[1,2,1],[1,1,2]])

def test_c2e_solve_relations():
    b = fmm.two.Cube([0,0], 1.0)
    s = fmm.two.surrounding_surface(5)
    cfg = fmm.two.FMMConfig(3.0, 1.1, 5, "laplaceD2", [])
    op = np.array(fmm.two.c2e_solve(s, b, 3.0, 1.1, cfg))

    for i in range(5):
        b2 = fmm.two.Cube(np.random.rand(2).tolist(), 1.0)
        op2 = np.array(fmm.two.c2e_solve(s, b2, 3.0, 1.1, cfg))
        np.testing.assert_almost_equal(op2, op)

    scale = 2.0
    b2 = fmm.two.Cube([0,0], scale)
    op2 = np.array(fmm.two.c2e_solve(s, b2, 3.0, 1.1, cfg)) / scale
    np.testing.assert_almost_equal(op2, op)
