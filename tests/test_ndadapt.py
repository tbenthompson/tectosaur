from tectosaur.ndadapt import *

def make_integrator(p, f):
    ps = [p] * 3
    q_unmapped = tensor_gauss(ps)

    def integrator(mins, maxs):
        out = []
        for i in range(mins.shape[0]):
            q = map_to(q_unmapped, mins[i,:], maxs[i,:])
            out.append(np.sum(f(q[0]) * q[1]))
        return out
    return integrator

def test_tensor_gauss():
    nqs = (4, 3, 6)
    q = tensor_gauss(nqs)
    for i in range(2 * nqs[0] - 1):
        for j in range(2 * nqs[1] - 1):
            for k in range(2 * nqs[2] - 1):
                f = lambda pts: (pts[:,0] ** i) * (pts[:,1] ** j) * (pts[:,2] ** k)
                est = sum(f(q[0]) * q[1])
                correct = np.prod([
                    (1.0 ** (idx + 1) - (-1.0) ** (idx + 1)) / (float(idx) + 1)
                    for idx in (i, j, k)
                ])
                np.testing.assert_almost_equal(est, correct)

def test_map_to():
    nqs = (4, 3, 6)
    q = map_to(tensor_gauss(nqs), np.array([0, 0, 0.]), np.array([1, 1, 1.]))
    i, j, k = 6, 4, 8
    f = lambda pts: (pts[:,0] ** i) * (pts[:,1] ** j) * (pts[:,2] ** k)
    est = sum(f(q[0]) * q[1])
    correct = np.prod([1.0 ** (idx + 1) / (float(idx) + 1) for idx in (i, j, k)])
    np.testing.assert_almost_equal(est, correct)

def make_integrator(d, p, f):
    ps = [p] * d
    q_unmapped = tensor_gauss(ps)

    def integrator(mins, maxs):
        out = []
        for i in range(mins.shape[0]):
            q = map_to(q_unmapped, mins[i,:], maxs[i,:])
            fvs = f(q[0])
            if len(fvs.shape) == 1:
                fvs = fvs[:,np.newaxis]
            assert(fvs.shape[0] == q[1].shape[0])
            out.append(np.sum(fvs * q[1][:,np.newaxis], axis = 0))
        return np.array(out)
    return integrator

def adapt_tester(f, correct):
    p = 5
    integrator = make_integrator(3, p, f)
    res, count = hadapt(integrator, (0,0,0), (1,1,1), 1e-14)
    np.testing.assert_almost_equal(res, correct)

def test_simple():
    adapt_tester(lambda pts: pts[:,0], 0.5)

def test_vector_simple():
    adapt_tester(lambda pts: np.array([pts[:,0], pts[:,1]]).T, [0.5, 0.5])

def test_harder_vector_integrals():
    adapt_tester(
        lambda pts: np.array([np.cos(pts[:,0]), np.sin(pts[:,1])]).T,
        [np.sin(1), 1 - np.cos(1)]
    )

def test_1dintegral():
    integrator = make_integrator(1, 5, lambda pts: np.sin(100 * pts[:,0]))
    res, count = hadapt(integrator, (0,), (1,), 1e-14)
    np.testing.assert_almost_equal(res, (1 - np.cos(100)) / 100.0)

def test_2dintegral():
    integrator = make_integrator(2, 5, lambda pts: np.sin(20 * pts[:,0] * pts[:,1]))
    res, count = hadapt(integrator, (0,0), (1,1), 1e-14)
    np.testing.assert_almost_equal(res, 0.1764264058805085268)

def test_limit_depth():
    res, count = hadapt(
        make_integrator(1, 5, lambda pts: np.sin(1000 * pts[:,0])),
        (0,), (1,), 1e-4,
        max_refinements = 2
    )
    assert(count == 7)

def test_negative_value():
    # Checks that the calculation of iguess is accounting for potentially negative
    # values of the integrals.
    res, count = hadapt(
        make_integrator(1, 5, lambda pts: -1e10 * pts[:,0] ** 4),
        (0,), (1,), 1e-14
    )
    assert(count == 3)
    np.testing.assert_almost_equal(res, -0.2 * 1e10)
