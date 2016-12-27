import os
import numpy as np
import matplotlib.pyplot as plt

import tectosaur.quadrature as quad
import tectosaur.geometry as geometry
import tectosaur.standardize as standardize
import tectosaur.limit as limit

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

def co_integrals(K, tri, rho_order, theta_order, tol, eps_start, n_steps, sm, pr):
    epsvs = eps_start * (2.0 ** (-np.arange(n_steps)))
    vals = []
    for eps in epsvs:
        rho_gauss = quad.gaussxw(rho_order)
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        theta_q = quad.gaussxw(theta_order)
        vals.append(adaptive_integrate.integrate_coincident(
            K, tri, tol, eps, sm, pr,
            rho_q[0].tolist(), rho_q[1].tolist(),
            theta_q[0].tolist(), theta_q[1].tolist()
        ))
    vals = np.array(vals)
    return epsvs, vals

def co_limit(K, tri, rho_order, theta_order, tol, eps_start, n_steps,
        eps_scale, sm, pr, include_log = False):
    epsvs, vals = co_integrals(
        K, tri, rho_order, theta_order, tol, eps_start * eps_scale, n_steps, sm, pr
    )
    return np.array([
        limit.limit(epsvs / eps_scale, vals[:, i], include_log) for i in range(81)
    ]), vals

def adj_integrals(K, obs_tri, src_tri, rho_order, theta_order, tol, eps_start, n_steps, sm, pr):
    epsvs = eps_start * (2.0 ** (-np.arange(n_steps)))
    vals = []
    for eps in epsvs:
        rho_gauss = quad.gaussxw(rho_order)
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        vals.append(adaptive_integrate.integrate_adjacent(
            K, obs_tri, src_tri, tol, eps,
            sm, pr, rho_q[0].tolist(), rho_q[1].tolist()
        ))

    vals = np.array(vals)
    return epsvs, vals

def adj_limit(K, obs_tri, src_tri, rho_order, theta_order, tol, eps_start, n_steps,
        eps_scale, sm, pr, include_log = False):
    epsvs, vals = adj_integrals(
        K, obs_tri, src_tri, rho_order, theta_order, tol, eps_start * eps_scale, n_steps, sm, pr
    )
    return np.array([
        limit.limit(epsvs / eps_scale, vals[:, i], include_log) for i in range(81)
    ])

def standardized_tri_tester(K, sm, pr, rho_order, theta_order, tol, eps_start, n_steps, tri):
    include_log = True
    standard_tri, labels, translation, R, scale = standardize.standardize(
        np.array(tri), 20, True
    )
    is_flipped = not (labels[1] == ((labels[0] + 1) % 3))

    np.testing.assert_almost_equal(
        standard_tri,
        [[0,0,0],[1,0,0],[standard_tri[2][0],standard_tri[2][1],0]]
    )

    correct_full, individual = co_limit(
        K, tri, rho_order, theta_order, tol, eps_start, n_steps, scale, sm, pr, include_log
    )
    correct_full = correct_full[:,0].reshape((3,3,3,3))

    # 1) calculate the standardized integrals
    epsvs, standard_vals = co_integrals(
        K, standard_tri, rho_order, theta_order, tol, eps_start * scale ** 2, n_steps, 1.0, pr
    )

    # 2) convert them to the appropriate values for true triangles
    unstandardized_vals = np.array([
        np.array(standardize.transform_from_standard(
            # standard_vals[i,:], K, sm, labels, translation, R, scale
            standard_vals[i,:], K, sm, labels, translation, R, scale
        )).reshape(81)
        for i in range(standard_vals.shape[0])
    ])

    # 3) take the limit in true space, not standardized space
    unstandardized = np.array([
        limit.limit(epsvs / (scale ** 2), unstandardized_vals[:, i], include_log)
        for i in range(81)
    ])[:,0].reshape((3,3,3,3))

    A = unstandardized[0,0,0,0]
    B = correct_full[0,0,0,0]

    print(
        str(tol) +
        " " + str(eps_start) +
        " " + str(n_steps) +
        " " + str(A) +
        " " + str(B)
    )
    err = np.abs((unstandardized[:,0,:,0] - correct_full[:,0,:,0]) / np.max(np.abs(correct_full[:,0,:,0])))
    assert(np.all(err < 0.03))
    # np.testing.assert_almost_equal(unstandardized, correct_full, 4)

def kernel_properties_tester(K, sm, pr):
    test_tris = [
        # [[0.0,0.0,0.0], [1.0,0.0,0.0], [0.4,0.3,0.0]], #NO TRANSFORMATION
        [[0.0,0.0,0.0], [2.0,0.0,0.0], [0.8,0.6,0.0]], # JUST SCALE
        [[0.0,0.0,0.0], [0.0,1.0,0.0], [-0.3,0.4,0.0]], # JUST ROTATE
        [[0.0,0.0,0.0], [0.0,1.1,0.0], [-0.3,0.4,0.0]], # ROTATE + SCALE
        [[0.0,0.0,0.0], [0.0,0.0,1.1], [0.0,-0.3,0.4]], # ROTATE + SCALE
        [[0.0,0.0,0.0], [0.0,0.3,1.1], [0.0,-0.3,0.4]],
        [[0.0,0.0,0.0], [0.0,-0.3,0.4], [0.0,0.35,1.1]],
        [[0.0,0.35,1.1], [0.0,0.0,0.0], [0.0,-0.3,0.4]],
        [[0.0, -0.3, 0.4], [0.0,0.35,1.1], [0.0,0.0,0.0]],
        [[1.0,0.0,0.0], [0.0,-0.3,0.45], [0.0,0.35,1.1]]
    ]
    for t in test_tris:
        print("TESTING " + str(t))
        standardized_tri_tester(K, sm, pr, 50, 50, 0.005, 0.08, 3, t)
        print("SUCCESS")

    # n_checks = 10
    # while True:
    #     tri = np.random.rand(3,3).tolist()

    #     try:
    #         test_tri(tri)
    #         print("SUCCESS!")
    #     except Exception as e:
    #         # Exception implies the triangle was malformed (angle < 20 degrees)
    #         continue

    #     n_checks -= 1
    #     if n_checks == 0:
    #         break

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

if __name__ == '__main__':
    test_vert_adj()
    # test_edge_adj()
