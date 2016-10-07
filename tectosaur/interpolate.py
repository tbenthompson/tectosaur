import numpy as np

def to_interval(a, b, x):
    return a + (b - a) * (x + 1.0) / 2.0

def from_interval(a, b, x):
    return ((x - a) / (b - a)) * 2.0 - 1.0

"""Chebyshev nodes in the [a,b] interval. Good for interpolation. Avoids Runge phenomenon."""
def cheb(a, b, n):
    out = []
    for i in range(n):
        out.append(to_interval(a, b, np.cos(((2 * i + 1) * np.pi) / (2 * n))))
    return out

def cheb_wts(a, b, n):
    j = np.arange(n)
    return ((-1) ** j) * np.sin(((2 * j + 1) * np.pi) / (2 * n))

def cheblob(a, b, n):
    out = []
    for i in range(n):
        out.append(to_interval(a, b, np.cos((i * np.pi) / (n - 1))))
    return out

def cheblob_wts(a, b, n):
    wts = ((-1) ** np.arange(n)) * np.ones(n)
    wts[0] *= 0.5
    wts[-1] *= 0.5
    return wts

def barycentric_eval(pts, wts, vals, x_hat):
    denom = 0.0
    numer = 0.0

    for p,w,v in zip(pts, wts, vals):
        p = np.where(np.abs(x_hat - p) < 1e-15, p + np.finfo(float).eps, p)
        kernel = w / np.prod(x_hat - p)
        denom += kernel
        numer += kernel * v
    return numer / denom

def barycentric_evalnd(pts, wts, vals, xhat):
    dist_list = []
    for d in range(pts.shape[1]):
        dist = np.tile(xhat[:,d,np.newaxis], (1,pts.shape[0]))\
            - np.tile(pts[:,d], (xhat.shape[0],1))

        dist[dist == 0] = np.finfo(float).eps
        dist_list.append(dist)

    K = wts / np.prod(np.array(dist_list), axis = 0)
    denom = np.sum(K, axis = 1)
    denom = np.where(denom != 0, denom, np.finfo(float).eps)
    interp_vals = (K.dot(vals) / denom)
    return interp_vals

if __name__ == '__main__':
    # test_barycentric_interp()
    # test_barycentric_interp2d()
    test_barycentric_interp3d()
