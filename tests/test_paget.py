import numpy as np
from tectosaur.util.paget import *
from tectosaur.util.quadrature import *
from tectosaur.nearfield.triangle_rules import vertex_interior_quad

def test_bk():
    for a in [1,2]:
        for n in range(a, 10):
            v1 = calc_bk(n, a)
            v2 = calc_bk2(n, a)
            np.testing.assert_almost_equal(v1, v2)

def test_ek():
    for a in [1,2]:
        for n in range(1, 10):
            bk = calc_bk(n, a)
            gx, gw = gaussxw(n)
            np.testing.assert_almost_equal(calc_e(bk, gx), calc_e2(bk, gx))

def test_order2_a1():
    qx, qw = map_to(paget(2, 1), [0, 1])
    np.testing.assert_almost_equal(sum(qw), 0)

def test_order2_a2():
    qx, qw = map_to(paget(2, 2), [0, 1])
    np.testing.assert_almost_equal(sum(qw), -1)

def test_cos_a1():
    f = lambda x: np.cos(x)
    exact = -0.23981174200056474
    for n in range(1, 13):
        q = map_to(paget(n, 1), [0, 1])
        res = quadrature(f, q)
    np.testing.assert_almost_equal(res, exact)

def test_cos_a2():
    f = lambda x: np.cos(x)
    exact = -1.48638536036015
    for n in range(1, 13):
        q = map_to(paget(n, 2), [0, 1])
        res = quadrature(f, q)
    np.testing.assert_almost_equal(res, exact)

def test_vertex_interior_quad():
    q = vertex_interior_quad(5, 5, True)
    R = np.linalg.norm(q[0], axis = 1)
    np.testing.assert_almost_equal(np.sum(q[1] / (R ** 2)), 0.0)

def test_vertex_interior_quad2():
    q = vertex_interior_quad(8, 5, False)
    R = np.linalg.norm(q[0], axis = 1)
    np.testing.assert_almost_equal(np.sum(q[1]), 0.5)
