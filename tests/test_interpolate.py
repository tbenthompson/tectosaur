import numpy as np
from tectosaur.nearfield.interpolate import *

def test_barycentric_1d():
    pts = np.array(cheblob(-1, 1, 3))[:,np.newaxis]
    wts = np.array(cheblob_wts(-1, 1, 3))
    xs = np.linspace(-1, 1, 20)[:,np.newaxis]
    f = lambda x: x ** 2
    vals = f(pts)
    result = barycentric_evalnd(pts, wts, vals, xs, np.float64)
    np.testing.assert_almost_equal(result, xs ** 2)

def ptswts3d(N):
    pts1d = cheb(-1,1,N)
    X,Y,Z = np.meshgrid(pts1d, pts1d, pts1d)
    pts = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
    wts1d = cheb_wts(-1,1,N)
    wts = np.outer(wts1d, np.outer(wts1d, wts1d)).ravel()
    return pts.copy(), wts.copy()

def interp_tester_one_output_dim(test_pts):
    N = 10
    pts, wts = ptswts3d(N)

    f = lambda xs: np.sin(xs[:,0] ** 3 - xs[:,1] * xs[:,2])
    vals = f(pts)[:, np.newaxis]

    correct = f(test_pts)
    interp_vals = barycentric_evalnd(
        pts, wts, vals, test_pts,
        np.float64
    )
    log_err = np.log10(np.abs(np.max(correct - interp_vals.flatten())))
    assert(np.log10(np.abs(np.max(correct - interp_vals.flatten()))) < -3)

def pts_grid(N):
    xs = np.linspace(-1, 1, N)
    xe,ye,ze = np.meshgrid(xs, xs, xs)
    xhat = np.array([xe.ravel(), ye.ravel(), ze.ravel()]).T.copy()
    return xhat

def test_barycentric_interp3d():
    xhat = pts_grid(30)
    interp_tester_one_output_dim(xhat)

def test_barycentric_degeneracy():
    pts, _ = ptswts3d(10)
    interp_tester_one_output_dim(pts[:1,:])

def test_multiple_output_dims():
    pts, wts = ptswts3d(10)
    xhat = pts_grid(30)
    f = lambda xs: np.array([np.cos(xs[:,0]) * xs[:,1], np.sin(xs[:,2])]).T.copy()
    vals = f(pts)
    out = barycentric_evalnd(pts, wts, vals, xhat, np.float64)
    correct = f(xhat)
    np.testing.assert_almost_equal(correct, out)

