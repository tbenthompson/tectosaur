import numpy as np
from tectosaur.util.paget import *
from tectosaur.util.quadrature import *

def test_bk():
    for n in range(1, 10):
        v1 = calc_bk(n)
        v2 = calc_bk2(n)
        np.testing.assert_almost_equal(v1, v2)

def test_ek():
    for n in range(1, 10):
        bk = calc_bk(n)
        gx, gw = gaussxw(n)
        np.testing.assert_almost_equal(calc_e(bk, gx), calc_e2(bk, gx))

def test_order2():
    qx, qw = paget(2)
    np.testing.assert_almost_equal(sum(qw), 0)

def test_cos():
    f = lambda x: np.cos(x)
    exact = -0.23981174200056474
    for n in range(1, 13):
        q = paget(n)
        res = quadrature(f, q)
        print(res - exact)
    np.testing.assert_almost_equal(res, exact)
