from tectosaur.util.quadrature import *
from tectosaur.nearfield.limit import richardson_quad

import numpy as np

def test_richardson10():
    h = np.array([1.0, 0.1, 0.01])
    q = richardson_quad(h, False, lambda h: (np.array([0.0]), np.array([1.0])))
    est = np.sum((h ** 2) * q[1])
    np.testing.assert_almost_equal(est, 0.0)

def test_log_richardson():
    h = 2.0 ** -np.arange(5)
    q = richardson_quad(h, True, lambda h: (np.array([0.0]), np.array([1.0])))
    vals = np.log(h) + h ** 2
    est = np.sum(vals * q[1])
    np.testing.assert_almost_equal(est, 0.0)

def test_gauss():
    est = quadrature(lambda x: x ** 7, map_to(gaussxw(4), [0, 1]))
    exact = 1.0 / 8.0
    np.testing.assert_almost_equal(est, exact)

def test_gauss2d_tri1():
    q = gauss2d_tri(2)
    result = quadrature(lambda x: 1, q)
    np.testing.assert_almost_equal(result, 0.5)

def test_gauss2d_tri2():
    q = gauss2d_tri(5)
    result = quadrature(lambda x: x[:,0] ** 3 * x[:,1] ** 4, q)
    np.testing.assert_almost_equal(result, 1.0 / 2520.0, 12)

def test_gauss2d_tri3():
    q = gauss2d_tri(7)
    result = quadrature(lambda x: np.sin(np.exp(x[:,0] * x[:,1] * 5)), q)
    np.testing.assert_almost_equal(result, 0.426659055902, 4)

def test_gauss2d_tri_using_symmetric_rules():
    q = gauss2d_tri(3)
    assert(q[0].shape[0] == 7)

def test_gauss4d_tri():
    q = gauss4d_tri(3, 3)
    result = quadrature(lambda x: 1, q)
    np.testing.assert_almost_equal(result, 0.25)

    result = quadrature(lambda x: (x[:,0] * x[:,1] * x[:,2] * x[:,3]) ** 2, q)
    np.testing.assert_almost_equal(result, 1.0 / (180.0 ** 2), 10)
