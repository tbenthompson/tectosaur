import numpy as np
from pycuda import driver as drv

import tectosaur.triangle_rules as triangle_rules
import tectosaur.integral_op as integral_op
import tectosaur.quadrature as quad
import tectosaur.mesh as mesh

from test_decorators import slow, golden_master
from laplace import laplace

@golden_master
def test_farfield_two_tris():
    pts = np.array(
        [[1, 0, 0], [2, 0, 0], [1, 1, 0],
        [5, 0, 0], [6, 0, 0], [5, 1, 0]]
    )
    obs_tris = np.array([[0, 1, 2]], dtype = np.int)
    src_tris = np.array([[3, 4, 5]], dtype = np.int)
    out = integral_op.farfield(1.0, 0.25, pts, obs_tris, src_tris, 3)
    return out

@golden_master
def test_gpu_edge_adjacent():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,-1,0],[2,0,0]]).astype(np.float32)
    obs_tris = np.array([[0,1,2]]).astype(np.int32)
    src_tris = np.array([[1,0,3]]).astype(np.int32)
    out = integral_op.edge_adj(8, 1.0, 0.25, pts, obs_tris, src_tris)
    return out

@golden_master
def test_gpu_vert_adjacent():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,-1,0],[2,0,0]]).astype(np.float32)
    obs_tris = np.array([[1,2,0]]).astype(np.int32)
    src_tris = np.array([[1,3,4]]).astype(np.int32)
    out = integral_op.vert_adj(3, 1.0, 0.25, pts, obs_tris, src_tris)
    return out

@golden_master
def test_coincident_gpu():
    n = 4
    w = 4
    pts, tris = mesh.rect_surface(n, n, [[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]])
    out = integral_op.coincident(8, 1.0, 0.25, pts, tris)
    return out

@golden_master
def test_full_integral_op():
    m = mesh.rect_surface(5, 5, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    out = integral_op.SelfIntegralOperator(5, 5, 5, 1.0, 0.25, m[0], m[1])
    np.random.seed(100)
    return out.dot(np.random.rand(out.shape[1]))

tri_ref = [[0,0,0],[1,0,0],[0,1,0]]
tri_down = [[1,0,0],[0,0,0],[0,-1,0]]
tri_up_right = [[0,1,0],[1,0,0],[1,1,0]]
def test_weakly_singular_adjacent():
    # When one of the basis functions is zero along the edge where the triangles
    # touch, the integral is no longer hypersingular, but instead is only weakly
    # singular.
    weakly_singular = [(2, 2), (2, 0), (2, 1), (0, 2), (1, 2)]
    q1 = triangle_rules.edge_adj_quad(0, 4, 4, 4, 4, True)
    q2 = triangle_rules.edge_adj_quad(0, 5, 5, 5, 5, True)
    for i, j in weakly_singular:
        K = lambda pts: laplace(tri_ref, tri_down, i, j, 0, pts)
        v = quad.quadrature(K, q1)
        v2 = quad.quadrature(K, q2)
        assert(np.abs((v - v2) / v2) < 0.012)

@slow
def test_edge_adj_quad():
    # The complement of the nonsingular and weakly singular sets above.
    # These basis function pairs are both non-zero along the triangle boundary
    hypersingular = [(0, 0), (0, 1), (1, 0), (1, 1)]
    eps = 0.01
    q1 = triangle_rules.edge_adj_quad(eps, 8, 8, 8, 8, False)
    q2 = triangle_rules.edge_adj_quad(eps, 9, 9, 9, 9, False)
    for i, j in hypersingular:
        K = lambda pts: laplace(tri_ref, tri_down, i, j, eps, pts)
        v = quad.quadrature(K, q1)
        v2 = quad.quadrature(K, q2)
        np.testing.assert_almost_equal(v, v2, 3)

@slow
def test_cancellation():
    result = []
    for n_eps in range(1,4):
        eps = 8 ** (-np.arange(n_eps).astype(np.float) - 1)
        qc = quad.richardson_quad(
            eps, lambda e: triangle_rules.coincident_quad(e, 15, 15, 15, 20)
        )
        qa = quad.richardson_quad(
            eps, lambda e: triangle_rules.edge_adj_quad(e, 15, 15, 15, 20, False)
        )

        Kco = lambda pts: laplace(tri_ref, tri_ref, 1, 1, pts[:,4], pts)
        Kadj_down = lambda pts: laplace(tri_ref, tri_down, 1, 0, pts[:,4], pts)
        tri_ref_rotated = [tri_ref[1], tri_ref[2], tri_ref[0]]
        Kadj_up_right = lambda pts: laplace(
                tri_ref_rotated, tri_up_right, 0, 1, pts[:,4], pts
        )
        Ic = quad.quadrature(Kco, qc)
        Ia1 = quad.quadrature(Kadj_down, qa)
        Ia2 = quad.quadrature(Kadj_up_right, qa)
        nondivergent = Ic + Ia1 + Ia2
        result.append(nondivergent)
    assert(abs(result[2] - result[1]) < 0.5 * abs(result[1] - result[0]))

def test_adjacent_rule():
    nq = 7
    q = triangle_rules.vertex_adj_quad(nq, nq, nq)
    est = quad.quadrature(lambda p: p[:,0]*p[:,1]*p[:,2]*p[:,3], q)
    correct = 1.0 / 576.0
    np.testing.assert_almost_equal(est, correct)

def test_coincident_simple():
    eps = 0.01

    q = triangle_rules.coincident_quad(0.01, 10, 10, 10, 10)

    result = quad.quadrature(lambda x: x[:, 2], q)
    np.testing.assert_almost_equal(result, 1.0 / 12.0, 2)

    result = quad.quadrature(lambda x: x[:, 3], q)
    np.testing.assert_almost_equal(result, 1.0 / 12.0, 2)

    result = quad.quadrature(lambda x: x[:, 2] * x[:, 3], q)
    np.testing.assert_almost_equal(result, 1.0 / 48.0, 2)

@slow
def test_coincident_laplace():
    tri = [
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]

    eps = 0.01
    for i in range(3):
        for j in range(i):
            K = lambda pts: laplace(tri, tri, i, j, eps, pts)
            q_accurate = triangle_rules.coincident_quad(eps, 18, 15, 15, 18)
            exact = quad.quadrature(K, q_accurate)
            def tryit(n1,n2,n3,n4):
                q = triangle_rules.coincident_quad(eps, n1, n2, n3, n4)
                result = quad.quadrature(K, q)
                return np.abs((result - exact) / exact)
            assert(tryit(7,6,4,10) < 0.005)
            assert(tryit(14,9,7,10) < 0.0005)
            assert(tryit(13,12,11,13) < 0.00005)
