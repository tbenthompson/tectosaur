from tectosaur.limit import *

def test_poly_limit():
    xs = 2.0 ** -np.arange(4)
    vals = 1.0 + xs ** 1 + xs ** 3
    coeffs = limit_coeffs(xs, vals, 0)
    np.testing.assert_almost_equal(coeffs, [1.0, 1.0, 0.0, 1.0])

    at0 = limit(xs, vals, 0)
    np.testing.assert_almost_equal(at0, [1.0, 0.0])

    at0_2 = richardson_limit(2.0, vals)
    np.testing.assert_almost_equal(at0_2, 1.0)

def test_limit_log():
    npts = 3
    xs = 2.0 ** -np.arange(npts)
    vals = np.log(xs) + xs ** 1
    coeffs = limit_coeffs(xs, vals, 1)
    np.testing.assert_almost_equal(coeffs, [1.0, 0.0, 1.0])
