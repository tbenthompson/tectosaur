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
theta_order = 50
starting_eps = 0.001
n_eps = 1
tol = 1e-4
n_A = 2
n_B = 2
n_pr = 2

filename = (
    '%s_%i_%f_%i_%f_%i_%i_%i_coincidenttable.npy' %
    (K, rho_order, starting_eps, n_eps, tol, n_A, n_B, n_pr)
)
print(filename)

all_eps = starting_eps * 2.0 ** -np.arange(n_eps)
rho_gauss = quad.gaussxw(rho_order)

def eval(pt):
    Ahat,Bhat,prhat = pt
    # From symmetry and enforcing that the edge (0,0)-(1,0) is always longest,
    # A,B can be limited to (0,0.5)x(0,1).
    A = to_interval(0.0, 0.5, Ahat)
    B = to_interval(0.0, 1.0, Bhat)
    pr = to_interval(0.0, 0.5, prhat)

    tri = [[0,0,0],[1,0,0],[A,B,0.0]]

    start = time.time()
    integrals = []
    for eps in all_eps:
        print('running: ' + str((pt, eps)))
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        theta_q = quad.gaussxw(theta_order)
        res = adaptive_integrate.integrate_coincident(
            K, tri, tol, eps, 1.0, pr,
            rho_q[0].tolist(), rho_q[1].tolist(),
            theta_q[0].tolist(), theta_q[1].tolist()
        )
        print(res[0])
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

def test_f(results, eval_fnc, pts, wts):
    limits = np.empty((results.shape[0], results.shape[2]))
    for i in range(results.shape[0]):
        limits[i,:] = take_limits(results[i,:,:])
    P = np.random.rand(pts.shape[1]) * 2 - 1.0
    correct = take_limits(np.array(eval_fnc(P)))
    for i in range(81):
        interp = barycentric_evalnd(pts, wts, limits[:,i], np.array([P]))[0]
        # print("testing:  " + str(i) + "     " + str(
        #     (correct[i], interp, np.abs((correct[i] - interp) / correct[i]), correct[i] - interp)
        # ))

def build_tables(eval_fnc, pts, wts):
    pool = multiprocessing.Pool()
    results = np.array([eval_fnc(p) for p in pts.tolist()])
    np.save(filename, results)
    np.random.seed(15)
    for i in range(3):
        test_f(results, eval_fnc, pts, wts)

if __name__ == '__main__':
    print("")
    print("")
    print("")
    print("")
    print("GO")
    integrals = eval([0.0, 1.0, 0.5])
    print(limit(all_eps, integrals, True))
    print("DONE")
    print("")
    print("")
    print("")
    print("")
    import sys;sys.exit()


    Ahats = cheb(-1, 1, n_A)
    Bhats = cheb(-1, 1, n_B)
    prhats = cheb(-1, 1, n_pr)
    Ah,Bh,Nh = np.meshgrid(Ahats, Bhats,prhats)
    pts = np.array([Ah.ravel(),Bh.ravel(), Nh.ravel()]).T

    Awts = cheb_wts(-1,1,n_A)
    Bwts = cheb_wts(-1,1,n_B)
    prwts = cheb_wts(-1,1,n_pr)
    wts = np.outer(Awts,np.outer(Bwts,prwts)).ravel()

    build_tables(eval, pts, wts)
