import numpy as np

import tectosaur.nearfield.triangle_rules as triangle_rules
import tectosaur.nearfield.nearfield_op as nearfield_op
import tectosaur.nearfield.limit as limit

import tectosaur.ops.dense_integral_op as dense_integral_op
import tectosaur.ops.sparse_integral_op as sparse_integral_op
import tectosaur.ops.mass_op as mass_op

import tectosaur.util.quadrature as quad
import tectosaur.mesh.mesh_gen as mesh_gen

from tectosaur.interior import interior_integral

from tectosaur.util.test_decorators import slow, golden_master, kernel
from tectosaur.util.timer import Timer

from laplace import laplace

# import tectosaur, logging
# tectosaur.logger.setLevel(logging.ERROR)
from tectosaur.util.logging import setup_logger
logger = setup_logger(__name__)

float_type = np.float32

@golden_master()
def test_farfield_two_tris(request):
    pts = np.array(
        [[1, 0, 0], [2, 0, 0], [1, 1, 0],
        [5, 0, 0], [6, 0, 0], [5, 1, 0]]
    )
    obs_tris = np.array([[0, 1, 2]], dtype = np.int)
    src_tris = np.array([[3, 4, 5]], dtype = np.int)
    params = [1.0, 0.25]
    out = dense_integral_op.farfield_tris(
        'elasticH3', params, pts, obs_tris, src_tris, 3, float_type
    )
    return out

@golden_master()
def test_gpu_vert_adjacent(request):
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,-1,0],[2,0,0]]).astype(np.float32)
    obs_tris = np.array([[1,2,0]]).astype(np.int32)
    src_tris = np.array([[1,3,4]]).astype(np.int32)
    params = [1.0, 0.25]
    out = nearfield_op.vert_adj(3, 'elasticH3', params, pts, obs_tris, src_tris, float_type)
    return out

def full_integral_op_tester(k, use_fmm, n = 5):
    pts = np.array([[0,0,0], [1,1,0], [0, 1, 1], [0,0,2]])
    tris = np.array([[0, 1, 2], [2, 1, 3]])
    rect_mesh = mesh_gen.make_rect(n, n, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    out = np.zeros(1)
    params = [1.0, 0.25]
    for m in [(pts, tris), rect_mesh]:
        dense_op = dense_integral_op.DenseIntegralOp(
            5, 3, 3, 2.0, k, params, m[0], m[1], float_type
        )
        x = np.ones(dense_op.shape[1])
        dense_res = dense_op.dot(x)
        if use_fmm:
            farfield_op_type = sparse_integral_op.FMMFarfieldBuilder(100, 3.0, 300)
        else:
            farfield_op_type = None
        sparse_op = sparse_integral_op.SparseIntegralOp(
            5, 3, 3, 2.0, k, params, m[0], m[1],
            float_type, farfield_op_type
        )
        sparse_res = sparse_op.dot(x)
        assert(np.max(np.abs(sparse_res - dense_res)) / np.mean(np.abs(dense_res)) < 5e-4)
        out = np.hstack((out, sparse_res))
    return out

@slow
@golden_master(digits = 5)
def test_full_integral_op_nofmm(request, kernel):
    return full_integral_op_tester(kernel, False)

@slow
@golden_master(digits = 7)
def test_full_integral_op_fmm(request):
    return full_integral_op_tester('elasticU3', True, n = 30)

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
    K = 'elasticH3'
    params = [1.0, 0.25]
    obs_tris = np.array([[0,1,2]])
    src_tris = np.array([[0,4,3]])
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[-0.5,0,0],[0,0,-2],[0.5,0.5,0]])

    nq = 6

    I = nearfield_op.vert_adj(nq, K, params, pts, obs_tris, src_tris, float_type)

    obs_basis_tris = np.array([
        [[0,0],[0.5,0.5],[0,1]], [[0,0],[1,0],[0.5,0.5]]
    ])
    src_basis_tris = np.array([
        [[0,0],[1,0],[0,1]], [[0,0],[1,0],[0,1]]
    ])
    obs_tris = np.array([[0,5,2], [0,1,5]])
    src_tris = np.array([[0,4,3], [0,4,3]])
    I0 = nearfield_op.vert_adj(nq, K, params, pts, obs_tris, src_tris, float_type)

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

    K = 'elasticH3'
    params = [1.0, 0.25]

    return interior_integral(obs_pts, obs_ns, (pts, tris), input, K, 4, 4, params, float_type)

@profile
def benchmark_nearfield_construction():
    corners = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    near_threshold = 1.5
    n = 80
    pts, tris = mesh_gen.make_rect(n, n, corners)
    n = nearfield_op.NearfieldIntegralOp(1, 1, 1, 2.0, 'elasticU3', [1.0, 0.25], pts, tris)

@profile
def benchmark_vert_adj():
    from tectosaur.util.timer import Timer
    import tectosaur.mesh.find_near_adj as find_near_adj
    from tectosaur.nearfield.pairs_integrator import PairsIntegrator
    kernel = 'elasticH3'
    params = [1.0, 0.25]
    float_type = np.float32
    L = 5
    nq_vert_adjacent = 7

    nx = ny = int(2 ** L / np.sqrt(2))
    t = Timer()
    pts, tris = mesh_gen.make_rect(nx, ny, [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])
    logger.debug('n_tris: ' + str(tris.shape[0]))
    t.report('make rect')
    close_or_touch_pairs = find_near_adj.find_close_or_touching(pts, tris, 1.25)
    nearfield_pairs, va, ea = find_near_adj.split_adjacent_close(close_or_touch_pairs, tris)
    t.report('find near')
    pairs_int = PairsIntegrator(kernel, params, float_type, 1, 1, pts, tris)
    t.report('setup integrator')
    va_mat_rot = pairs_int.vert_adj(nq_vert_adjacent, va)
    t.report('vert adj')

if __name__ == "__main__":
    benchmark_vert_adj()

