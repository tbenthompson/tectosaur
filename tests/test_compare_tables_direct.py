from tectosaur.dense_integral_op import DenseIntegralOp

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
