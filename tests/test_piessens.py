from tectosaur.util.piessens import *

def map_singular_pt(x0, a, b):
    return (2 * (x0 - a) / (b - a)) - 1

def test_piessen_neg_1_1():
    # Example 1 from Piessens
    f = lambda x: np.exp(x)
    exact = 2.11450175
    piessen_est = 2.11450172
    x, w = piessen_neg_one_to_one_nodes(2)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(piessen_est, est)

def test_piessen_0_1():
    # Example 1 from Piessens mapped to [0,1]
    g = lambda x: np.exp(x)
    f = lambda x: g((2 * x) - 1)
    exact = 2.11450175
    piessen_est = 2.11450172
    x, w = piessen_method(2, 0.0, 1.0, 0.5, False)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(piessen_est, est)

def test_piessen_0_1_with_singularity():
    # Example 1 from Piessens mapped to [0,1] and with singularity
    g = lambda x: np.exp(x) / x
    f = lambda x: 2 * g((2 * x) - 1)
    exact = 2.11450175
    piessen_est = 2.11450172
    x, w = piessen_method(2, 0.0, 1.0, 0.5)
    est = np.sum(f(x) * w)
    np.testing.assert_almost_equal(piessen_est, est)

def test_QuadOneOverR_1():
    f = lambda x: 1 / (x - 0.4)
    exact = np.log(3.0 / 2.0)
    mapped_x0 = map_singular_pt(0.4, 0.0, 1.0)
    x, w = piessens(2, mapped_x0, nonsingular_N = 10)
    qx, qw = map_to((x, w), (0.0, 1.0))
    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

def test_QuadOneOverR_2():
    # Example 1 from Piessens
    g = lambda x: np.exp(x) / x
    f = lambda x: 2 * g((2 * x) - 1)
    exact = 2.11450175
    mapped_x0 = map_singular_pt(0.5, 0.0, 1.0)
    x, w = piessens(8, mapped_x0, nonsingular_N = 10)
    qx, qw = map_to((x, w), (0.0, 1.0))

    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

def test_QuadOneOverR_3():
    # Example 2 from Piessens
    g = lambda x: np.exp(x) / (np.sin(x) - np.cos(x))
    f = lambda x: np.pi / 2.0 * g(np.pi / 2.0 * x)
    exact = 2.61398312
    # Piessens estimate derived with a two pt rule.
    piessens_est = 2.61398135

    mapped_x0 = map_singular_pt(0.5, 0.0, 1.0)
    x, w = piessens(2, mapped_x0)
    qx, qw = map_to((x, w), (0.0, 1.0))

    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(piessens_est, est)

# Tests x in the upper half of the interval
def test_QuadOneOverR_4():
    f = lambda x: np.exp(x) / (x - 0.8)
    exact = -1.13761642399

    mapped_x0 = map_singular_pt(0.8, 0.0, 1.0)
    x, w = piessens(2, mapped_x0, nonsingular_N = 20)
    qx, qw = map_to((x, w), (0.0, 1.0))
    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

# Tests x in the lower half of the interval.
def test_QuadOneOverR_5():
    f = lambda x: np.exp(x) / (x - 0.2)
    exact = 3.139062607254266

    mapped_x0 = map_singular_pt(0.2, 0.0, 1.0)
    x, w = piessens(2, mapped_x0, nonsingular_N = 50)
    qx, qw = map_to((x, w), (0.0, 1.0))

    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

def test_piessens_4_5():
    f = lambda x: np.exp(x - 4) / (x - 4.2)
    exact = 3.139062607254266

    mapped_x0 = map_singular_pt(4.2, 4.0, 5.0)
    x, w = piessens(2, mapped_x0, nonsingular_N = 8)
    qx, qw = map_to((x, w), (4.0, 5.0))

    est = np.sum(f(qx) * qw)
    np.testing.assert_almost_equal(exact, est)

def test_endpoint():
    f = lambda x: np.log(x)
    mapped_x0 = map_singular_pt(0.0, -1.0, 1.0)
    print(mapped_x0)
    x, w = piessens(4, mapped_x0, nonsingular_N = 4)
    qx, qw = map_to((x, w), (-1.0, 1.0))

    est = np.sum(f(qx) * qw)
    print(est)
    # np.testing.assert_almost_equal(exact, est)
