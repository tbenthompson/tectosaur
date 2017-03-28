from tectosaur.limit import *

def test_poly_limit():
    xs = 2.0 ** -np.arange(4)
    vals = 1.0 + xs ** 1 + xs ** 3
    coeffs = limit_coeffs(xs, vals, 0)[0]
    np.testing.assert_almost_equal(coeffs, [1.0, 1.0, 0.0, 1.0])

    at0 = limit(xs, vals, 0)
    np.testing.assert_almost_equal(at0, [1.0, 0.0])

    at0_2 = richardson_limit(2.0, vals)
    np.testing.assert_almost_equal(at0_2, 1.0)

def test_limit_log():
    npts = 3
    xs = 2.0 ** -np.arange(npts)
    vals = np.log(xs) + xs ** 1
    coeffs = limit_coeffs(xs, vals, 1)[0]
    np.testing.assert_almost_equal(coeffs, [1.0, 0.0, 1.0])

def test_limit_inv():
    npts = 3
    xs = 2.0 ** -np.arange(npts)
    vals = 1 / xs + xs ** 1
    coeffs = limit_coeffs(xs, vals, 0, True)[0]
    np.testing.assert_almost_equal(coeffs, [1.0, 0.0, 1.0])

def pi_seq(n):
    vals = [4]
    for i in range(1, n + 1):
        term = 4 * ((-1) ** i) / (2 * i + 1)
        vals.append(vals[-1] + term)
    return np.array(vals)

def test_aitken_pi():
    assert(np.abs(aitken(pi_seq(10), 5)[-1] - np.pi) < 1e-9)

def sqrt2_seq(n):
    vals = [1]
    for i in range(n):
        vals.append((vals[-1] + 2 / vals[-1]) / 2)
    return np.array(vals)

def test_aitken_sqrt():
    assert(np.abs(aitken(sqrt2_seq(5), 1)[-1] - np.sqrt(2)) < 1e-9)
