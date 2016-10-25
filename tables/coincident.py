import time
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

import tectosaur.quadrature as quad
import tectosaur.geometry as geometry
from tectosaur.interpolate import cheb, cheb_wts, barycentric_evalnd, to_interval
from tectosaur.limit import limit, richardson_limit

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

# These A, B limits come from ablims.py
minlegalA = 0.0939060848748 - 0.01
minlegalB = 0.233648154379 - 0.01
maxlegalB = 0.865565992417 + 0.01

# tol = 0.0001
# rho_order = 100
# starting_eps = 0.01
# n_eps = 7
# K = "U"

# tol = 1e-4
# rho_order = 100
# starting_eps = 1e-6
# n_eps = 3
# K = "T"
# K = "A"

tol = 1e-5
rho_order = 100
starting_eps = 1e-5
n_eps = 3
K = "H"

n_A = 8
n_B = 8
n_pr = 8

# play parameters
K = "H"
rho_order = 50
starting_eps = 0.08
n_eps = 4
tol = 0.01
n_A = 3
n_B = 3
n_pr = 3

filename = (
    '%s_%i_%f_%i_%f_%i_%i_%i_coincidenttable.npy' %
    (K, rho_order, starting_eps, n_eps, tol, n_A, n_B, n_pr)
)
print(filename)

all_eps = starting_eps * 2.0 ** -np.arange(n_eps)
rho_gauss = quad.gaussxw(rho_order)

Ahats = cheb(-1, 1, n_A)
Bhats = cheb(-1, 1, n_B)
prhats = cheb(-1, 1, n_pr)
Ah,Bh,Nh = np.meshgrid(Ahats, Bhats,prhats)
pts = np.array([Ah.ravel(),Bh.ravel(), Nh.ravel()]).T

Awts = cheb_wts(-1,1,n_A)
Bwts = cheb_wts(-1,1,n_B)
prwts = cheb_wts(-1,1,n_pr)
wts = np.outer(Awts,np.outer(Bwts,prwts)).ravel()

n_dims = 3

def eval(pt):
    Ahat,Bhat,prhat = pt
    A = to_interval(minlegalA, 0.5, Ahat)
    B = to_interval(minlegalB, maxlegalB, Bhat)
    pr = to_interval(0.0, 0.5, prhat)

    tri = [[0,0,0],[1,0,0],[A,B,0.0]]
    eps_scale = np.sqrt(np.linalg.norm(geometry.tri_normal(tri)))

    start = time.time()
    integrals = []
    for eps in all_eps:
        print('running: ' + str((pt, eps)))
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        res = adaptive_integrate.integrate_coincident(
            K, tri, tol, eps * eps_scale, 1.0, pr,
            rho_q[0].tolist(), rho_q[1].tolist()
        )
        integrals.append(res)
    return integrals

def take_limits(integrals):
    out = np.empty(81)
    remove_divergence = False
    for i in range(81):
        out[i] = limit(all_eps, integrals[:, i], remove_divergence)
    if not remove_divergence:
        np.testing.assert_almost_equal(out, richardson_limit(2.0, integrals))
    return out

def test_f(input):
    seed, results = input
    limits = np.empty((results.shape[0], results.shape[2]))
    for i in range(results.shape[0]):
        limits[i,:] = take_limits(results[i,:,:])
    np.random.seed(seed)
    pt = np.random.rand(n_dims) * 2 - 1.0
    correct = take_limits(np.array(eval(pt)))
    for i in range(81):
        interp = barycentric_evalnd(pts, wts, limits[:,i], np.array([pt]))[0]
        print("testing:  " + str(i) + "     " + str(
            (correct[i], interp, np.abs((correct[i] - interp) / correct[i]), correct[i] - interp)
        ))

# results = np.load(filename)
# for i in range(12):
#     test_f((i, results))

pool = multiprocessing.Pool()
results = np.array(pool.map(eval, pts.tolist()))
np.save(filename, results)
pool.map(test_f, zip(range(12), [results for i in range(12)]))
