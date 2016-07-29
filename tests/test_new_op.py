from tectosaur.dense_taylor_integral_op import *
from tectosaur.mesh import *

def test_offset_pts():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]]).astype(np.float64)
    tris = np.array([[0,1,2],[1,3,2]])
    out = calc_near_obs_pts(pts, tris, [], [], gauss2d_tri(1), 0.5)
    np.testing.assert_almost_equal(out, [[[0.25, 0.5, 0.5]], [[0.5, 0.75, 0.5]]])

    pts = np.array([[0,0,0],[10,0,0],[0,10,0],[10,10,0]]).astype(np.float64)
    out = calc_near_obs_pts(pts, tris, [], [], gauss2d_tri(1), 0.5)
    np.testing.assert_almost_equal(out, [[[2.5, 5, 5]], [[5, 7.5, 5]]])

def test_full_op():
    m = rect_surface(30, 30, [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]])
    iop = DenseTaylorIntegralOp(0.5, 5, 15, 15, 10, 7, 3.0, 3, 1.0, 0.25, m[0], m[1])
    print(iop)
