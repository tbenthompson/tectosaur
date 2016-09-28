from tectosaur.quadrature import *

import numpy as np

def test_gauss():
    est = quadrature(lambda x: x ** 7, map_to(gaussxw(4), [0, 1]))
    exact = 1.0 / 8.0
    np.testing.assert_almost_equal(est, exact)

def test_richardson():
    hs = 2 ** np.linspace(2, -2, 5)
    xs = hs ** 4
    est = richardson(hs, xs)
    np.testing.assert_almost_equal(est[-1][-1], 0.0)

def test_richardson_quad():
    gauss_q = gaussxw(11)
    q = richardson_quad(
        2 ** np.linspace(0, -5, 5),
        lambda e: map_to(sinh_transform(gauss_q, -1, e), [0, 1])
    )
    f = lambda x: 2 * x[:, 1] / (x[:, 0] ** 2 + x[:, 1] ** 2)
    res = quadrature(f, q)
    np.testing.assert_almost_equal(np.pi, res, 4)

def test_richardson10():
    h = np.array([1.0, 0.1, 0.01])
    q = richardson_quad(h, lambda h: (np.array([0.0]), np.array([1.0])))
    est = np.sum((h ** 2) * q[1])
    np.testing.assert_almost_equal(est, 0.0)

def test_sinh():
    eps = 0.01
    q = map_to(sinh_transform(gaussxw(12), -1, eps), [0, 1])
    est = quadrature(lambda x: 1.0 / (x ** 2 + eps ** 2), q)
    exact = 156.079666010823330
    np.testing.assert_almost_equal(est, exact, 3)

def test_aimi_diligenti():
    q = aimi_diligenti(gaussxw(12), 7, 7)
    est = quadrature(lambda x: np.log(1 - x) * np.log(x + 1), q)
    exact = -1.10155082811
    np.testing.assert_almost_equal(est, exact, 6)

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
