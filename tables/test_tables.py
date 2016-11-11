import numpy as np
import matplotlib.pyplot as plt

import tectosaur.quadrature as quad
import tectosaur.geometry as geometry
from tectosaur.dense_integral_op import DenseIntegralOp
import tectosaur.standardize as standardize
import tectosaur.limit as limit

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

from gpu_integrator import new_integrate_coincident

def co_integrals(K, tri, rho_order, tol, eps_start, n_steps, sm, pr, new_method = False):
    epsvs = eps_start * (2.0 ** (-np.arange(n_steps)))
    vals = []
    for eps in epsvs:
        rho_gauss = quad.gaussxw(rho_order)
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        if new_method:
            vals.append(new_integrate_coincident(
                K, tri, tol, eps, sm, pr, rho_q[0].tolist(), rho_q[1].tolist()
            ))
        else:
            vals.append(adaptive_integrate.integrate_coincident(
                K, tri, tol, eps, sm, pr, rho_q[0].tolist(), rho_q[1].tolist()
            ))
    vals = np.array(vals)
    return epsvs, vals

def co_limit(K, tri, rho_order, tol, eps_start, n_steps,
        eps_scale, sm, pr, include_log = True, new_method = False):
    epsvs, vals = co_integrals(
        K, tri, rho_order, tol, eps_start * eps_scale, n_steps, sm, pr, new_method
    )
    return np.array([
        limit.limit(epsvs / eps_scale, vals[:, i], include_log) for i in range(81)
    ])

def adj_integrals(K, obs_tri, src_tri, rho_order, tol, eps_start, n_steps, sm, pr):
    epsvs = eps_start * (2.0 ** (-np.arange(n_steps)))
    vals = []
    for eps in epsvs:
        rho_gauss = quad.gaussxw(rho_order)
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        vals.append(adaptive_integrate.integrate_adjacent(
            K, obs_tri, src_tri, tol, eps, sm, pr, rho_q[0].tolist(), rho_q[1].tolist()
        ))
    vals = np.array(vals)
    return epsvs, vals

def adj_limit(K, obs_tri, src_tri, rho_order, tol, eps_start, n_steps,
        eps_scale, sm, pr, include_log = True):
    epsvs, vals = adj_integrals(
        K, obs_tri, src_tri, rho_order, tol, eps_start * eps_scale, n_steps, sm, pr
    )
    return np.array([
        limit.limit(epsvs / eps_scale, vals[:, i], include_log) for i in range(81)
    ])

def standardized_tri_tester(K, sm, pr, rho_order, tol, eps_start, n_steps, tri):
    standard_tri, labels, translation, R, scale = standardize.standardize(np.array(tri), 20)
    is_flipped = not (labels[1] == ((labels[0] + 1) % 3))

    np.testing.assert_almost_equal(
        standard_tri,
        [[0,0,0],[1,0,0],[standard_tri[2][0],standard_tri[2][1],0]]
    )

    correct_full = co_limit(
        K, tri, rho_order, tol, eps_start, n_steps, scale, sm, pr
    ).reshape((3,3,3,3))

    # 1) calculate the standardized integrals
    epsvs, standard_vals = co_integrals(
        K, standard_tri.tolist(), rho_order, tol, eps_start, n_steps, 1.0, pr
    )

    # 2) convert them to the appropriate values for true triangles
    unstandardized_vals = np.array([
        standardize.transform_from_standard(
            standard_vals[i,:].reshape((3,3,3,3)), K, sm, labels, translation, R, scale
        ).reshape(81)
        for i in range(standard_vals.shape[0])
    ])

    # 3) take the limit in true space, not standardized space
    true_epsvs = epsvs / scale
    unstandardized = np.array([
        limit.limit(true_epsvs, unstandardized_vals[:, i], True)
        for i in range(81)
    ]).reshape((3,3,3,3))

    print(
        str(tol) +
        " " + str(eps_start) +
        " " + str(n_steps) +
        " " + str(unstandardized[0,0,0,0])
    )
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

def runner(i):
    t = [[0,0,0],[1,0,0],[0.4,0.3,0]]
    standardized_tri_tester('H', 1.0, 0.25, 60, 0.01, 10 ** (-i), 3, t)
    return 0

def test_sing_removal_conv():
    # standardized_tri_tester('H', 1.0, 0.25, 60, 0.0001, 0.001, 3, t)
    # standardized_tri_tester('H', 1.0, 0.25, 60, 0.0001, 0.0001, 3, t)
    # standardized_tri_tester('H', 1.0, 0.25, 60, 0.0001, 0.0001, 4, t)
    # standardized_tri_tester('H', 1.0, 0.25, 80, 0.0001, 0.0001, 4, t)
    import multiprocessing
    p = multiprocessing.Pool()
    runner(2)
    p.map(runner, [2,3,4,5,6,7])

def test_coincident():
    K = 'H'
    eps = 0.08
    pts = np.array([[0,0,0],[1,0,0],[0.4,0.3,0]])
    eps_scale = np.sqrt(np.linalg.norm(geometry.tri_normal(pts)))
    tris = np.array([[0,1,2]])

    op = DenseIntegralOp(
        [eps, eps / 2], 17, 10, 13, 10, 10, 3.0,
        K, 1.0, 0.25, pts, tris, remove_sing = True
    )

    res = co_limit(
        K, pts[tris[0]].tolist(), 100, 0.001, eps, 2, eps_scale,
        1.0, 0.25, include_log = True
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
    eps_scale = np.sqrt(np.linalg.norm(geometry.tri_normal(pts[tris[0]])))
    op = DenseIntegralOp(
        [eps, eps / 2], 10, 15, 10, 10, 10, 3.0, K, 1.0, 0.25, pts, tris,
        remove_sing = True
    )

    res = adj_limit(
        K, pts[tris[0]].tolist(), pts[tris[1]].tolist(), 100, 0.001,
        eps, 2, eps_scale, 1.0, 0.25, include_log = True
    )
    np.testing.assert_almost_equal(res, op.mat[:9,9:].reshape(81), 4)

def test_new_mthd_coincident():
    K = 'H'
    eps = 0.08
    pts = np.array([[0,0,0],[1,0,0],[0.4,0.3,0]])
    eps_scale = np.sqrt(np.linalg.norm(geometry.tri_normal(pts)))
    tris = np.array([[0,1,2]])
    res_new = co_limit(
        K, pts[tris[0]].tolist(), 100, 0.01, eps, 2, eps_scale,
        1.0, 0.25, include_log = True, new_method = True
    )
    res_old = co_limit(
        K, pts[tris[0]].tolist(), 100, 0.01, eps, 2, eps_scale,
        1.0, 0.25, include_log = True
    )
    np.testing.assert_almost_equal(res_new, res_old, 3)

if __name__ == '__main__':
    test_vert_adj()
    # test_edge_adj()
