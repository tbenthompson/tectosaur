"""
Multidimensional adaptive quadrature
http://ab-initio.mit.edu/wiki/index.php/Cubature <-- using this
http://mint.sbg.ac.at/HIntLib/
http://www.feynarts.de/cuba/

is hcubature or pcubature better? answer: doesn't seem to matter much but probably p

problem! running out of memory for higher accuracy integrals.. how to fix? can i split up
the domain of integration? how do i reduce the accuracy requirements after splitting a domain?
i suspect the answer is buried in the cubature code...
another idea for reducing memory requirements by a factor of 81 is to split up the integrals so that each dimension and basis func is computed separately. or just split by basis func since different basis function pairs should need very different sets of points -- some will be zero in different places than others

should i used vector integrands or is it better to split up the integral into each
individual component so that they don't all need to take as long as the most expensive one
(DONE)test out the richardson process

(DONE) start with 0,0-1,0-0,1 and try all the rotation and scaling stuff.
(DONE) make sure i can get 6 digits of accuracy reasonably easily for all kernels
(DONE) relabeling

TODO:
can i explicitly model and remove the divergence? projects/archive/analytical_bem_integrals/experiments/calc_quad.py has the start of some code to do this based on a modified interpolation basis for the extrapolation process.

write the table lookup procedure -- multidimensional barycentric lagrange interpolation

test the lookup by selecting random legal triangles and comparing the interpolation
"""

import time
from math import cos, pi
import numpy as np
import matplotlib.pyplot as plt

import cppimport
adaptive_integrator = cppimport.imp('adaptive_integrator')

import standardize

def to_interval(a, b, x):
    return a + (b - a) * (x + 1.0) / 2.0

"""Chebyshev nodes in the [a,b] interval. Good for interpolation. Avoids Runge phenomenon."""
def cheb(a, b, n):
    out = []
    for i in range(n):
        out.append(to_interval(a, b, cos(((2 * i + 1) * pi) / (2 * n))))
    return out

def cheblob(a, b, n):
    out = []
    for i in range(n):
        out.append(to_interval(a, b, cos((i * pi) / (n - 1))))
    return out

def cheblob_wts(a, b, n):
    wts = [0.5]
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

def test_barycentric_interp():
    pts = cheblob(-1,1,5)
    wts = cheblob_wts(-1,1,5)
    vals = [0,1,0,1,0]
    xs = np.linspace(-1,1,100)
    interp_vals = [barycentric_eval(pts, wts, vals, x) for x in xs]
    np.testing.assert_almost_equal(interp_vals[3], 0.41489441539529159)
    # plt.plot(pts, vals, 'o')
    # plt.plot(xs, interp_vals)
    # plt.show()

def test_barycentric_interp2d():
    for N in range(3, 40, 1):
        pts1d = cheblob(-1,1,N)
        X,Y = np.meshgrid(pts1d, pts1d)
        pts = np.array([X.flatten(), Y.flatten()]).T
        wts1d = cheblob_wts(-1,1,N)
        wts = np.outer(wts1d, wts1d).flatten()
        vals = np.sin((pts[:,0] + pts[:,1] - 1.0) * 5)

        ne = 100
        xs = np.linspace(-1, 1, ne)
        xe,ye = np.meshgrid(xs, xs)

        xdist = np.tile(xe.flatten()[:,np.newaxis], (1,pts.shape[0]))\
            - np.tile(pts[:,0], (xe.size,1))

        ydist = np.tile(ye.flatten()[:,np.newaxis], (1,pts.shape[0]))\
            - np.tile(pts[:,1], (ye.size,1))

        xdist[xdist == 0] = np.finfo(float).eps
        ydist[ydist == 0] = np.finfo(float).eps

        K = wts / (ydist * xdist)
        denom = np.sum(K, axis = 1)
        denom = np.where(denom != 0, denom, np.finfo(float).eps)
        interp_vals = (K.dot(vals) / denom).reshape((ne,ne))
        plt.imshow(interp_vals)
        plt.title(N)
        plt.colorbar()
        plt.show()

test_barycentric_interp()
test_barycentric_interp2d()

def richardson_limit(step_ratio, values):
    n_steps = len(values)
    last_level = values
    this_level = None

    for m in range(1, n_steps):
        this_level = []
        for i in range(n_steps - m):
            mult = step_ratio ** m
            factor = 1.0 / (mult - 1.0)
            low = last_level[i]
            high = last_level[i + 1]
            moreacc = factor * (mult * high - low)
            this_level.append(moreacc)
        last_level = this_level
    return this_level[0]

def calc(k_name, tri, tol, start_eps, eps_step, n_steps, sm, pr):
    eps = start_eps
    vals = []
    for i in range(n_steps):
        res = np.array(adaptive_integrator.integrate(K, tri, tol, eps, sm, pr))
        vals.append(res)
        eps /= eps_step
    return richardson_limit(eps_step, vals)


# These A, B limits come from ablims.py
minlegalA = 0.0939060848748 - 0.01
minlegalB = 0.233648154379 - 0.01
maxlegalB = 0.865565992417 + 0.01

tol = 0.01
eps_start = 0.01
eps_step = 2.0
n_steps = 2
pr = 0.25
sm = 1000.0
K = "U"

n_A = 3
# n_B = 3
As = chebab(minlegalA, 0.5, n_A)
# Bs = chebab(minlegalB, maxlegalB, n_B)
res = []
for A in As:
    res.append(calc(
        K, [[0,0,0],[1,0,0],[A,0.4,0.0]], tol, eps_start, eps_step, n_steps, 1.0, pr
    ))
print(res)

def test_tri(tri):
    standard_tri, labels, R, factor = standardize.standardize(np.array(tri))
    np.testing.assert_almost_equal(
        standard_tri,
        [[0,0,0],[1,0,0],[standard_tri[2][0],standard_tri[2][1],0]]
    )
    print(standard_tri)

    correct_full = calc(
        K, tri, tol, eps_start, eps_step, n_steps, sm, pr
    ).reshape((3,3,3,3))

    standard_full = calc(
        K, standard_tri.tolist(), tol, eps_start, eps_step, n_steps, 1.0, pr
    ).reshape((3,3,3,3))

    for sb1 in range(3):
        for sb2 in range(3):
            standardized = R.T.dot(standard_full[sb1,:,sb2,:]).dot(R) / (sm * factor ** 3)
            cb1 = labels[sb1]
            cb2 = labels[sb2]
            correct = correct_full[cb1,:,cb2,:]
            np.testing.assert_almost_equal(correct, standardized, 4)

    # np.testing.assert_almost_equal(correct, standardized, 4)

# tri = [[0.0,0.0,0.0], [1.1,0.0,0.0], [0.4,0.3,0.0]]
# tri = [[0.0,0.0,0.0], [0.0,1.1,0.0], [-0.3,0.4,0.0]]
# tri = [[0.0,0.0,0.0], [0.0,0.0,1.1], [0.0,-0.3,0.4]]
# tri = [[0.0,0.0,0.0], [0.0,0.3,1.1], [0.0,-0.3,0.4]]

# tri = [[0.0,0.0,0.0], [0.0,-0.3,0.4], [0.0,0.35,1.1]]
# test_tri(tri)
# tri = [[0.0,0.35,1.1], [0.0,0.0,0.0], [0.0,-0.3,0.4]]
# test_tri(tri)
# tri = [[0.0, -0.3, 0.4], [0.0,0.35,1.1], [0.0,0.0,0.0]]
# test_tri(tri)
# tri = [[1.0,0.0,0.0], [0.0,-0.3,0.45], [0.0,0.35,1.1]]
# test_tri(tri)

n_checks = 10
while True:
    tri = np.random.rand(3,3).tolist()

    try:
        test_tri(tri)
    except Exception as e:
        continue

    n_checks -= 1
    if n_checks == 0:
        break
