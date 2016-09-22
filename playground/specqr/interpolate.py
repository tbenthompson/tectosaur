import numpy as np

def to_interval(a, b, x):
    return a + (b - a) * (x + 1.0) / 2.0

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

#a vectorized matlab version of this: https://www.mathworks.com/matlabcentral/fileexchange/5511-2d-barycentric-lagrange-interpolation/content/barylag2d.m
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

def test_barycentric_interp():
    pts = cheb(-1,1,5)
    wts = cheb_wts(-1,1,5)
    vals = [0,1,0,1,0]
    xs = np.linspace(-1,1,100)
    interp_vals = [barycentric_eval(pts, wts, vals, x) for x in xs]
    import matplotlib.pyplot as plt
    plt.plot(pts, vals, 'o')
    plt.plot(xs, interp_vals)
    plt.show()

def test_barycentric_interp2d():
    import matplotlib.pyplot as plt
    for N in range(3, 40, 1):
        pts1d = cheblob(-1,1,N)
        X,Y = np.meshgrid(pts1d, pts1d)
        pts = np.array([X.ravel(), Y.ravel()]).T
        wts1d = cheblob_wts(-1,1,N)
        wts = np.outer(wts1d, wts1d).ravel()

        f = lambda xs: np.sin((xs[:,0] + xs[:,1] - 1.0) * 5)
        vals = f(pts)


        ne = 100
        xs = np.linspace(-1, 1, ne)
        xe,ye = np.meshgrid(xs, xs)
        xhat = np.array([xe.ravel(), ye.ravel()]).T
        interp_vals = barycentric_evalnd(pts, wts, vals, xhat).reshape((ne,ne))

        plt.imshow(interp_vals)
        plt.title(N)
        plt.colorbar()
        plt.show()

def test_barycentric_interp3d():
    import matplotlib.pyplot as plt
    for N in range(5,18,2):
        pts1d = cheb(-1,1,N)
        X,Y,Z = np.meshgrid(pts1d, pts1d, pts1d)
        pts = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
        wts1d = cheb_wts(-1,1,N)
        wts = np.outer(wts1d, np.outer(wts1d, wts1d)).ravel()

        f = lambda xs: np.sin(xs[:,0] ** 3 - xs[:,1] * xs[:,2])
        vals = f(pts)

        ne = 30
        xs = np.linspace(-1, 1, ne)
        xe,ye,ze = np.meshgrid(xs, xs, xs)
        xhat = np.array([xe.ravel(), ye.ravel(), ze.ravel()]).T
        correct = f(xhat)
        interp_vals = barycentric_evalnd(pts, wts, vals, xhat).reshape((ne,ne,ne))
        print(np.max(correct - interp_vals.flatten()))

    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    # test_barycentric_interp()
    # test_barycentric_interp2d()
    test_barycentric_interp3d()
