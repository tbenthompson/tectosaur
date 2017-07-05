import numpy as np

import tectosaur.nearfield.triangle_rules as triangle_rules
import tectosaur.nearfield.vert_adj as nearfield_op
import tectosaur.nearfield.limit as limit

# import tectosaur.ops.fmm_integral_op as fmm_integral_op
import tectosaur.ops.dense_integral_op as dense_integral_op
import tectosaur.ops.sparse_integral_op as sparse_integral_op
import tectosaur.ops.mass_op as mass_op

import tectosaur.util.quadrature as quad

import tectosaur.mesh.mesh_gen as mesh_gen
import tectosaur.mesh.find_nearfield as find_nearfield
import tectosaur.mesh.adjacency as adjacency

from tectosaur.interior import interior_integral

from tectosaur.util.test_decorators import slow, golden_master, kernel

from laplace import laplace

import tectosaur, logging
tectosaur.logger.setLevel(logging.ERROR)

def test_nearfield():
    corners = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    pts, tris = mesh_gen.make_rect(3,3 ,corners)
    assert(tris.shape[0] == 8)
    va, ea = adjacency.find_adjacents(tris)
    near_pairs = find_nearfield.find_nearfield(pts, tris, va, ea, 2.5)
    check_for = [
        (0, 5), (0, 6), (0, 3), (1, 7), (2, 7), (3, 0),
        (4, 7), (5, 0), (6, 0), (7, 2), (7, 1), (7, 4)
    ]
    assert(len(near_pairs) == len(check_for))
    for pair in check_for:
        assert(pair in near_pairs)

@golden_master()
def test_farfield_two_tris(request):
    pts = np.array(
        [[1, 0, 0], [2, 0, 0], [1, 1, 0],
        [5, 0, 0], [6, 0, 0], [5, 1, 0]]
    )
    obs_tris = np.array([[0, 1, 2]], dtype = np.int)
    src_tris = np.array([[3, 4, 5]], dtype = np.int)
    params = [1.0, 0.25]
    out = dense_integral_op.farfield('elasticH', params, pts, obs_tris, src_tris, 3)
    return out

@golden_master()
def test_gpu_edge_adjacent(request):
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,-1,0],[2,0,0]]).astype(np.float32)
    obs_tris = np.array([[0,1,2]]).astype(np.int32)
    src_tris = np.array([[1,0,3]]).astype(np.int32)
    params = [1.0, 0.25]
    out = nearfield_op.edge_adj(
        8, [0.1, 0.01], 'elasticH', params, pts, obs_tris, src_tris, remove_sing = False
    )
    return out

@golden_master()
def test_gpu_vert_adjacent(request):
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,-1,0],[2,0,0]]).astype(np.float32)
    obs_tris = np.array([[1,2,0]]).astype(np.int32)
    src_tris = np.array([[1,3,4]]).astype(np.int32)
    params = [1.0, 0.25]
    out = nearfield_op.vert_adj(3, 'elasticH', params, pts, obs_tris, src_tris)
    return out

@golden_master(5)
def test_coincident_gpu(request):
    n = 4
    w = 4
    pts, tris = mesh_gen.make_rect(n, n, [[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]])
    params = [1.0, 0.25]
    out = nearfield_op.coincident(
        8, [0.1, 0.01], 'elasticH', params, pts, tris, remove_sing = False
    )
    return out

def full_integral_op_tester(k):
    pts = np.array([[0,0,0], [1,1,0], [0, 1, 1], [0,0,2]])
    tris = np.array([[0, 1, 2], [2, 1, 3]])
    rect_mesh = mesh_gen.make_rect(5, 5, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    out = np.zeros(1)
    params = [1.0, 0.25]
    for m in [(pts, tris), rect_mesh]:
        dense_op = dense_integral_op.DenseIntegralOp(
            [0.1, 0.01], 5, 5, 5, 3, 3, 3.0, k, params, m[0], m[1]
        )
        np.random.seed(100)
        x = np.random.rand(dense_op.shape[1])
        dense_res = dense_op.dot(x)
        sparse_op = sparse_integral_op.SparseIntegralOp(
            [0.1, 0.01], 5, 5, 5, 3, 3, 3.0, k, params, m[0], m[1]
        )
        sparse_res = sparse_op.dot(x)
        assert(np.max(np.abs(sparse_res - dense_res)) < 2e-6)
        out = np.hstack((out, sparse_res))
    return out

@slow
@golden_master(digits = 5)
def test_full_integral_op(request, kernel):
    return full_integral_op_tester(kernel)

def check_simple(q, digits):
    est = quad.quadrature(lambda p: p[:,0]*p[:,1]*p[:,2]*p[:,3], q)
    correct = 1.0 / 576.0
    np.testing.assert_almost_equal(est, correct, digits)

    est = quad.quadrature(lambda p: p[:,0]**6*p[:,1]*p[:,3], q)
    correct = 1.0 / 3024.0
    np.testing.assert_almost_equal(est, correct, digits)

    est = quad.quadrature(lambda p: p[:,0]*p[:,2]**6*p[:,3], q)
    correct = 1.0 / 3024.0
    np.testing.assert_almost_equal(est, correct, digits)

def test_vertex_adjacent_simple():
    nq = 8
    q = triangle_rules.vertex_adj_quad(nq, nq, nq)
    check_simple(q, 7)


@slow
def test_coincident_laplace():
    tri = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])

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
            assert(tryit(19,13,17,16) < 0.000005)

def test_mass_op():
    m = mesh_gen.make_rect(2, 2, [[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
    op = mass_op.MassOp(3, m[0], m[1])
    exact00 = quad.quadrature(
        lambda x: (1 - x[:,0] - x[:,1]) * (1 - x[:,0] - x[:,1]),
        quad.gauss2d_tri(10)
    )
    exact03 = quad.quadrature(
        lambda x: (1 - x[:,0] - x[:,1]) * x[:,0],
        quad.gauss2d_tri(10)
    )
    np.testing.assert_almost_equal(op.mat[0,0], exact00)
    np.testing.assert_almost_equal(op.mat[0,3], exact03)

def test_vert_adj_separate_bases():
    K = 'elasticH'
    params = [1.0, 0.25]
    obs_tris = np.array([[0,1,2]])
    src_tris = np.array([[0,4,3]])
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[-0.5,0,0],[0,0,-2],[0.5,0.5,0]])

    nq = 6

    I = nearfield_op.vert_adj(nq, K, params, pts, obs_tris, src_tris)

    obs_basis_tris = np.array([
        [[0,0],[0.5,0.5],[0,1]], [[0,0],[1,0],[0.5,0.5]]
    ])
    src_basis_tris = np.array([
        [[0,0],[1,0],[0,1]], [[0,0],[1,0],[0,1]]
    ])
    obs_tris = np.array([[0,5,2], [0,1,5]])
    src_tris = np.array([[0,4,3], [0,4,3]])
    I0 = nearfield_op.vert_adj(nq, K, params, pts, obs_tris, src_tris)

    from tectosaur.nearfield.table_lookup import fast_lookup
    I1 = np.array([fast_lookup.sub_basis(
        I0[i].flatten().tolist(), obs_basis_tris[i].tolist(), src_basis_tris[i].tolist()
    ) for i in range(2)]).reshape((2,3,3,3,3))
    np.testing.assert_almost_equal(I[0], I1[0] + I1[1], 6)

@golden_master()
def test_interior(request):
    np.random.seed(10)
    corners = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    pts, tris = mesh_gen.make_rect(3,3 ,corners)
    obs_pts = pts.copy()
    obs_pts[:,2] += 1.0
    obs_ns = np.random.rand(*obs_pts.shape)
    obs_ns /= np.linalg.norm(obs_ns, axis = 1)[:,np.newaxis]

    input = np.ones(tris.shape[0] * 9)

    K = 'elasticH'
    params = [1.0, 0.25]

    return interior_integral(obs_pts, obs_ns, (pts, tris), input, K, 4, 4, params)


# def test_fmm_integral_op():
#     np.random.seed(13)
#     m = mesh_gen.make_sphere([0,0,0], 1, 1)
#     args = [
#         [], 1, 1, 5, 3, 4, 3.0,
#         'U', 1.0, 0.25, m[0], m[1], True
#     ]
#     op = fmm_integral_op.FMMIntegralOp(*args)
#     op2 = sparse_integral_op.SparseIntegralOp(*args)
#
#     v = np.random.rand(op.shape[1])
#     a = op.farfield_dot(v)
#     b = op2.farfield_dot(v)
#     np.testing.assert_almost_equal(a, b)
