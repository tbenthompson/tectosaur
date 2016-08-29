from tectosaur.dense_taylor_integral_op import *
from tectosaur.dense_integral_op import DenseIntegralOp
from tectosaur.mesh import *

def test_offset_pts():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]]).astype(np.float64)
    tris = np.array([[0,1,2],[1,3,2]])
    out = calc_near_obs_dir(pts, tris, [], [], gauss2d_tri(1))
    np.testing.assert_almost_equal(out, [[[0, 0, 1]], [[0, 0, 1]]])

    pts = np.array([[0,0,0],[10,0,0],[0,10,0],[10,10,0]]).astype(np.float64)
    out = calc_near_obs_dir(pts, tris, [], [], gauss2d_tri(1))
    np.testing.assert_almost_equal(out, [[[0, 0, 10]], [[0, 0, 10]]])

def test_full_op():
    m = rect_surface(2, 2, [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    for nq in range(30,60,10):
        print(nq)
        iop = DenseTaylorIntegralOp(0.02, nq, 200, 15, 10, 7, 3.0, 3, 1e0, 0.25, m[0], m[1])
        print(iop.mat[:3,:3])
    print(7.24634826e-02)
    # eps = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.0125 / 2.0]
    # iop2 = DenseIntegralOp(eps, 20, 20, 6, 3, 6, 4.0, 'U', 1e0, 0.25, m[0], m[1])
    # print(iop2.mat[:3,:3])
    # import ipdb; ipdb.set_trace()
    # np.testing.assert_almost_equal(iop.mat, iop2.mat, 2)
