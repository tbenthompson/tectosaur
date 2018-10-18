import tectosaur as tct
from tectosaur.util.quadrature import *
from tectosaur.nearfield.limit import richardson_quad
from tectosaur.nearfield.triangle_rules import *
from tectosaur.util.test_decorators import slow
from tectosaur.nearfield.nearfield_op import PairsIntegrator

import numpy as np

def test_richardson10():
    h = np.array([1.0, 0.1, 0.01])
    q = richardson_quad(h, False, lambda h: (np.array([0.0]), np.array([1.0])))
    est = np.sum((h ** 2) * q[1])
    np.testing.assert_almost_equal(est, 0.0)

def test_log_richardson():
    h = 2.0 ** -np.arange(5)
    q = richardson_quad(h, True, lambda h: (np.array([0.0]), np.array([1.0])))
    vals = np.log(h) + h ** 2
    est = np.sum(vals * q[1])
    np.testing.assert_almost_equal(est, 0.0)

def test_gauss():
    est = quadrature(lambda x: x ** 7, map_to(gaussxw(4), [0, 1]))
    exact = 1.0 / 8.0
    np.testing.assert_almost_equal(est, exact)

def test_gauss2d_tri1():
    q = gauss2d_tri(2)
    result = quadrature(lambda x: 1, q)
    np.testing.assert_almost_equal(result, 0.5)

def test_gauss2d_tri2():
    q = gauss2d_tri(5)
    result = quadrature(lambda x: x[:,0] ** 3 * x[:,1] ** 4, q)
    np.testing.assert_almost_equal(result, 1.0 / 2520.0, 12)

def test_gauss2d_tri3():
    q = gauss2d_tri(7)
    result = quadrature(lambda x: np.sin(np.exp(x[:,0] * x[:,1] * 5)), q)
    np.testing.assert_almost_equal(result, 0.426659055902, 4)

def test_gauss2d_tri_using_symmetric_rules():
    q = gauss2d_tri(3)
    assert(q[0].shape[0] == 7)

def test_gauss4d_tri():
    q = gauss4d_tri(3, 3)
    result = quadrature(lambda x: 1, q)
    np.testing.assert_almost_equal(result, 0.25)

    result = quadrature(lambda x: (x[:,0] * x[:,1] * x[:,2] * x[:,3]) ** 2, q)
    np.testing.assert_almost_equal(result, 1.0 / (180.0 ** 2), 10)

def check_simple(q, digits):
    est = quadrature(lambda p: 1.0, q)
    np.testing.assert_almost_equal(est, 0.25, digits)

    est = quadrature(lambda p: p[:,0]*p[:,1]*p[:,2]*p[:,3], q)
    correct = 1.0 / 576.0
    np.testing.assert_almost_equal(est, correct, digits)

    est = quadrature(lambda p: p[:,0]**6*p[:,1]*p[:,3], q)
    correct = 1.0 / 3024.0
    np.testing.assert_almost_equal(est, correct, digits)

    est = quadrature(lambda p: p[:,0]*p[:,1]*p[:,3]**6, q)
    correct = 1.0 / 1344.0
    np.testing.assert_almost_equal(est, correct, digits)

    est = quadrature(lambda p: p[:,0]*p[:,2]**6*p[:,3], q)
    correct = 1.0 / 3024.0
    np.testing.assert_almost_equal(est, correct, digits)

def test_vertex_adjacent_simple():
    nq = 8
    q = vertex_adj_quad(nq, nq, nq)
    check_simple(q, 7)

def test_vert_adj_real():
    def f(p):
        x2 = -p[:,2]
        y2 = -p[:,3]
        r = np.sqrt((p[:,0] - x2) ** 2 + (p[:,1] - y2) ** 2)
        return 1.0 / r

    est = [quadrature(lambda p: f(p), vertex_adj_quad(nq, nq, nq)) for nq in [9, 10]]
    np.testing.assert_almost_equal(est[0], est[1])

def test_coincident_sauter_simple():
    nq = 10
    q = coincident_quad(nq)
    check_simple(q, 7)

def test_coincident_real():
    def f(p):
        r = np.sqrt((p[:,0] - p[:,2]) ** 2 + (p[:,1] - p[:,3]) ** 2)
        return 1.0 / r

    est = [quadrature(lambda p: f(p), coincident_quad(nq)) for nq in [9, 10]]
    np.testing.assert_almost_equal(est[0], est[1])

def test_edge_adj_simple():
    nq = 6
    q = edge_adj_quad(nq)
    check_simple(q, 7)

def test_edge_adj_real():
    def f(p):
        y2 = -p[:,3]
        r = np.sqrt((p[:,0] - p[:,2]) ** 2 + (p[:,1] - y2) ** 2)
        return 1.0 / r

    est = [quadrature(lambda p: f(p), edge_adj_quad(nq)) for nq in [9, 10]]
    np.testing.assert_almost_equal(est[0], est[1])

def test_edge_adj_nan_problem_09_24_18():
    pts = np.array([[416615.84,5083954.,-24137.959],
        [411611.34,5084030.5,-23061.324],[414169.56,5086354.5,-23601.914],
        [418575.,5086595.5,-24544.795]],dtype=np.float32)
    tris = np.array([[2, 3, 0], [2, 0, 1]]).astype(np.int32)

    pairs_int = PairsIntegrator(
        'elasticU3', [1.0, 0.25], np.float32, 2, 5, pts, tris
    )

    ea = get_ea(pts, tris)
    mat = pairs_int.edge_adj(9, ea)
    print(mat)

def get_ea(pts, tris):
    import tectosaur.mesh.find_near_adj as find_near_adj
    from tectosaur.nearfield.nearfield_op import (
        to_tri_space,
        resolve_ea_rotation)
    obs_subset = np.arange(tris.shape[0])
    src_subset = np.arange(tris.shape[0])
    close_or_touch_pairs = find_near_adj.find_close_or_touching(
        pts, tris[obs_subset], pts, tris[src_subset], 2.0
    )
    nearfield_pairs_dofs, va_dofs, ea_dofs = find_near_adj.split_adjacent_close(
        close_or_touch_pairs, tris[obs_subset], tris[src_subset]
    )
    ea = resolve_ea_rotation(to_tri_space(ea_dofs, obs_subset, src_subset))
    return ea

def convergence_tester(f):
    n = 10
    corners = [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]]
    pts, tris = tct.make_rect(n, n, corners)
    pts = np.array([
        (0,0,0),
        (1,0,0),
        (0,1,0),
        (0,-1,0)
    ])
    #TODO: deal with flipping
    tris = np.array([[0,1,2],[1,0,3]])
    Ks = [
        'elasticU3',
        'elasticRT3',
        'elasticRA3',
        'elasticRH3'
    ]
    for K in Ks:
        for i in range(5):
            print(K)
        pairs_int = PairsIntegrator(
            K, [1.0, 0.25], np.float64, 2, 5, pts, tris
        )

        for nq in range(1, 13, 2):
            mat1 = f(nq, pairs_int, pts, tris)
            mat2 = f(nq + 1, pairs_int, pts, tris)
            diff = mat1 - mat2
            err = np.abs(diff / mat2)
            err[np.isnan(err)] = 1e-15

            l2_diff =  np.sqrt(np.sum(diff ** 2))
            l2_mat = np.sqrt(np.sum(mat2 ** 2))
            print(nq, mat1.flatten()[0], mat2.flatten()[0], l2_diff / l2_mat)

def convergence_coincident():
    def f(nq, pairs_int, pts, tris):
        co_tris = np.arange(tris.shape[0])
        co_indices = np.array([co_tris, co_tris]).T.copy()
        return pairs_int.coincident(nq, co_indices)
    convergence_tester(f)

def convergence_edgeadj():
    def f(nq, pairs_int, pts, tris):
        import tectosaur.mesh.find_near_adj as find_near_adj
        from tectosaur.nearfield.nearfield_op import (
            to_tri_space,
            resolve_ea_rotation)
        return pairs_int.edge_adj(nq, get_ea(pts, tris))

    convergence_tester(f)

def convergence_vertadj():
    def f(nq, pairs_int, pts, tris):
        import tectosaur.mesh.find_near_adj as find_near_adj
        from tectosaur.nearfield.nearfield_op import (
            to_tri_space,
            resolve_ea_rotation)
        obs_subset = np.arange(tris.shape[0])
        src_subset = np.arange(tris.shape[0])
        close_or_touch_pairs = find_near_adj.find_close_or_touching(
            pts, tris[obs_subset], pts, tris[src_subset], 2.0
        )
        nearfield_pairs_dofs, va_dofs, ea_dofs = find_near_adj.split_adjacent_close(
            close_or_touch_pairs, tris[obs_subset], tris[src_subset]
        )
        va = to_tri_space(va_dofs, obs_subset, src_subset)
        return pairs_int.vert_adj(nq, va)
    convergence_tester(f)

if __name__ == "__main__":
    # convergence_coincident()
    convergence_edgeadj()
    # convergence_vertadj()
