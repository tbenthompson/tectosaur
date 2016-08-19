from tectosaur.dense_taylor_integral_op import *
# from tectosaur.dense_integral_op import DenseIntegralOp
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
    iop = DenseTaylorIntegralOp(0.5, 5, 15, 15, 10, 7, 3.0, 3, 1.0, 0.25, m[0], m[1])
    # iop2 = DenseIntegralOp([0.2, 0.1, 0.05, 0.025, 0.0125], 25, 25,
    print(iop)
