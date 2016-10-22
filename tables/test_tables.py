import numpy as np
import matplotlib.pyplot as plt

import tectosaur.quadrature as quad
import tectosaur.geometry as geometry
from tectosaur.interpolate import cheb, cheb_wts, barycentric_evalnd
from tectosaur.dense_integral_op import DenseIntegralOp
import tectosaur.standardize as standardize

import limit
import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

def test_barycentric_interp3d():
    for i, N in enumerate(range(5,18,2)):
        pts1d = cheb(-1,1,N)
        X,Y,Z = np.meshgrid(pts1d, pts1d, pts1d)
        pts = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
        wts1d = cheb_wts(-1,1,N)
        wts = np.outer(wts1d, np.outer(wts1d, wts1d)).ravel()

        f = lambda xs: np.sin(xs[:,0] ** 3 - xs[:,1] * xs[:,2])
        vals = f(pts)

        ne = 30
        xs = np.linspace(-1, 1, ne)
        xe,ye,ze = np.meshgrid(xs, xs, xs)
        xhat = np.array([xe.ravel(), ye.ravel(), ze.ravel()]).T
        correct = f(xhat)
        interp_vals = barycentric_evalnd(pts, wts, vals, xhat).reshape((ne,ne,ne))
        log_err = np.log10(np.max(correct - interp_vals.flatten()))
        print(log_err)
        assert(np.log10(np.max(correct - interp_vals.flatten())) < -(i+1))

# def calc_I(eps):
#     tri1 = [[0,0,0],[1,0,0],[0,1,0.0]]
#     tri2 = [[1,0,0],[0,0,0],[0,-1,0]]
#     tol = 1e-5
#     rho_order = 80
#
#     rho_gauss = quad.gaussxw(rho_order)
#     rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
#     return adaptive_integrate.integrate_coincident(
#         "H", tri1, tol, eps, 1.0, 0.25,
#         rho_q[0].tolist(), rho_q[1].tolist()
#     )[0]
#
#     # Basis 1 and 0 divergence should cancel between coincident and adjacent
#     # return adaptive_integrate.integrate_coincident(
#     #     "H", tri1, tol, eps, 1.0, 0.25,
#     #     rho_q[0].tolist(), rho_q[1].tolist()
#     # )[3]
#     # return adaptive_integrate.integrate_adjacent(
#     #     "H", tri1, tri2,
#     #     tol, eps, 1.0, 0.25, rho_q[0].tolist(), rho_q[1].tolist()
#     # )[0]
#
# # for starting_eps in [0.1,0.05,0.025,0.01,0.001]:
# def play(starting_eps, n_steps):
#     eps = [starting_eps]
#     vals = [calc_I(eps[0])]
#     print("START")
#     for i in range(n_steps):
#         eps.append(eps[-1] / 2.0)
#         vals.append(calc_I(eps[-1]))
#
#         terms = make_terms(i + 2, True)
#         mat = [[t(e) for t in terms] for e in eps]
#         print(np.linalg.cond(mat))
#         coeffs = np.linalg.solve(mat, vals)
#
#         print("log coeff: " + str(coeffs[0]))
#         result = coeffs[1]
#         print("extrap to 0: " + str(result))

def calc_integrals(K, tri, rho_order, tol, eps_start, n_steps, sm, pr):
    epsvs = eps_start * (2.0 ** (-np.arange(n_steps)))
    vals = []
    for eps in epsvs:
        rho_gauss = quad.gaussxw(rho_order)
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        vals.append(adaptive_integrate.integrate_coincident(
            K, tri, tol, eps, sm, pr, rho_q[0].tolist(), rho_q[1].tolist()
        ))
    vals = np.array(vals)
    return epsvs, vals

def calc_limit(K, tri, rho_order, tol, eps_start, n_steps, sm, pr):
    epsvs, vals = calc_integrals(K, tri, rho_order, tol, eps_start, n_steps, sm, pr)
    return np.array([limit.limit(epsvs, vals[:, i], True) for i in range(81)])

def standardized_tri_tester(K, sm, pr, rho_order, tol, eps_start, n_steps, tri):

    standard_tri, labels, translation, R, scale = standardize.standardize(np.array(tri), 20)
    is_flipped = not (labels[1] == ((labels[0] + 1) % 3))

    np.testing.assert_almost_equal(
        standard_tri,
        [[0,0,0],[1,0,0],[standard_tri[2][0],standard_tri[2][1],0]]
    )

    correct_full = calc_limit(
        K, tri, rho_order, tol, eps_start / scale, n_steps, sm, pr
    ).reshape((3,3,3,3))

    epsvs, standard_vals = calc_integrals(
        K, standard_tri.tolist(), rho_order, tol, eps_start, n_steps, 1.0, pr
    )
    unstandardized_vals = np.array([
        standardize.transform_from_standard(
            standard_vals[i,:].reshape((3,3,3,3)), K, sm, labels, translation, R, scale
        ).reshape(81)
        for i in range(standard_vals.shape[0])
    ])
    unstandardized = np.array(
        [limit.limit(epsvs / scale, unstandardized_vals[:, i], True) for i in range(81)]
    ).reshape((3,3,3,3))
    import ipdb; ipdb.set_trace()

    np.testing.assert_almost_equal(unstandardized, correct_full, 4)

def kernel_properties_tester(K, sm, pr):
    test_tris = [
        [[0.0,0.0,0.0], [1.1,0.0,0.0], [0.4,0.3,0.0]]
        ,[[0.0,0.0,0.0], [0.0,1.1,0.0], [-0.3,0.4,0.0]]
        ,[[0.0,0.0,0.0], [0.0,0.0,1.1], [0.0,-0.3,0.4]]
        ,[[0.0,0.0,0.0], [0.0,0.3,1.1], [0.0,-0.3,0.4]]
        ,[[0.0,0.0,0.0], [0.0,-0.3,0.4], [0.0,0.35,1.1]]
        ,[[0.0,0.35,1.1], [0.0,0.0,0.0], [0.0,-0.3,0.4]]
        ,[[0.0, -0.3, 0.4], [0.0,0.35,1.1], [0.0,0.0,0.0]]
        ,[[1.0,0.0,0.0], [0.0,-0.3,0.45], [0.0,0.35,1.1]]
    ]
    for t in test_tris:
        standardized_tri_tester(K, sm, pr, 50, 0.05, 0.08, 3, t)
        print("SUCCESS")

    n_checks = 10
    while True:
        tri = np.random.rand(3,3).tolist()

        try:
            test_tri(tri)
            print("SUCCESS!")
        except Exception as e:
            # Exception implies the triangle was malformed (angle < 20 degrees)
            continue

        n_checks -= 1
        if n_checks == 0:
            break

def test_U_properties():
    kernel_properties_tester('U', 1.0, 0.25)

def test_T_properties():
    kernel_properties_tester('T', 1.0, 0.25)

def test_A_properties():
    kernel_properties_tester('A', 1.0, 0.25)

def test_H_properties():
    kernel_properties_tester('H', 1.0, 0.25)

def test_coincident():
    K = 'H'
    eps = 0.01
    pts = np.array([[0,0,0],[1,0,0],[0,1,0]])
    tris = np.array([[0,1,2]])

    op = DenseIntegralOp([eps], 20, 10, 13, 10, 10, 3.0, K, 1.0, 0.25, pts, tris)

    rho_order = 100
    rho_gauss = quad.gaussxw(rho_order)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    res = adaptive_integrate.integrate_coincident(
        K, pts[tris[0]].tolist(), 0.001, eps, 1.0, 0.25,
        rho_q[0].tolist(), rho_q[1].tolist()
    )
    np.testing.assert_almost_equal(res, op.mat.reshape(81), 3)

def test_vert_adj():
    K = 'H'

    pts = np.array([[0,0,0],[1,0,0],[0.5,0.5,0],[0,1,0],[0,-1,0],[0.5,-0.5,0]])
    tris = np.array([[0,2,3],[0,4,1]])
    op = DenseIntegralOp([0.01], 10, 10, 13, 10, 10, 3.0, K, 1.0, 0.25, pts, tris)
    res = adaptive_integrate.integrate_no_limit(
        K, pts[tris[0]].tolist(), pts[tris[1]].tolist(), 0.0001, 1.0, 0.25
    )
    np.testing.assert_almost_equal(res, op.mat[:9,9:].reshape(81), 5)

def test_edge_adj():
    K = 'H'

    eps = 0.08
    pts = np.array([[0,0,0],[1,0,0],[0.5,0.5,0],[0,1,0],[0,-1,0],[0.5,-0.5,0]])
    tris = np.array([[0,1,2],[1,0,4]])
    op = DenseIntegralOp([eps], 10, 15, 10, 10, 10, 3.0, K, 1.0, 0.25, pts, tris)

    rho_order = 100
    tol = 0.005
    rho_gauss = quad.gaussxw(rho_order)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    res = adaptive_integrate.integrate_adjacent(
        K, pts[tris[0]].tolist(), pts[tris[1]].tolist(), tol, eps * np.sqrt(0.5), 1.0, 0.25,
        rho_q[0].tolist(), rho_q[1].tolist()
    )
    np.testing.assert_almost_equal(res, op.mat[:9,9:].reshape(81), 4)

def test_limit_log():
    npts = 3
    xs = 2.0 ** -np.arange(npts)
    vals = np.log(xs) + xs ** 1 + np.random.rand(npts) * 0.001
    coeffs = limit.limit_coeffs(xs, vals, False)
    print(coeffs)
    # np.testing.assert_almost_equal(coeffs, [1.0, 0.0, 1.0])

if __name__ == '__main__':
    test_vert_adj()
    # test_edge_adj()
