import numpy as np

from tectosaur.interpolate import cheb, cheb_wts, barycentric_evalnd

def ptswts3d(N):
    pts1d = cheb(-1,1,N)
    X,Y,Z = np.meshgrid(pts1d, pts1d, pts1d)
    pts = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
    wts1d = cheb_wts(-1,1,N)
    wts = np.outer(wts1d, np.outer(wts1d, wts1d)).ravel()
    return pts, wts

def test_barycentric_interp3d():
    for i, N in enumerate(range(5,18,2)):
        pts, wts = ptswts3d(N)

        f = lambda xs: np.sin(xs[:,0] ** 3 - xs[:,1] * xs[:,2])
        vals = f(pts)

        ne = 30
        xs = np.linspace(-1, 1, ne)
        xe,ye,ze = np.meshgrid(xs, xs, xs)
        xhat = np.array([xe.ravel(), ye.ravel(), ze.ravel()]).T
        correct = f(xhat)
        interp_vals = barycentric_evalnd(pts, wts, vals, xhat).reshape((ne,ne,ne))
        log_err = np.log10(np.max(correct - interp_vals.flatten()))
        print(log_err)
        assert(np.log10(np.max(correct - interp_vals.flatten())) < -(i+1))
