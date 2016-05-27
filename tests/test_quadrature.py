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

def test_gauss4d():
    q = gauss4d_tri(2)
    result = quadrature(lambda x: 1, q)
    np.testing.assert_almost_equal(result, 0.25)

