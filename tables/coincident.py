import time
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

import tectosaur.quadrature as quad

from interpolate import cheb, cheb_wts, barycentric_evalnd, to_interval
from limit import limit

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

tol = 0.0001
rho_order = 100
starting_eps = 0.0001
n_eps = 4
K = "H"

n_A = 8
n_B = 8
n_pr = 8

# play parameters
# K = "H"
# rho_order = 80
# starting_eps = 0.1
# n_eps = 3
# tol = 0.01
# n_A = 3
# n_B = 3
# n_pr = 3

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

def eval(pt):
    Ahat,Bhat,prhat = pt
    A = to_interval(minlegalA, 0.5, Ahat)
    B = to_interval(minlegalB, maxlegalB, Bhat)
    pr = to_interval(0.0, 0.5, prhat)

    tri = [[0,0,0],[1,0,0],[A,B,0.0]]

    start = time.time()
    integrals = []
    for eps in all_eps:
        print('running: ' + str((pt, eps)))
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        res = adaptive_integrate.integrate_coincident(
            K, tri, tol, eps, 1.0, pr,
            rho_q[0].tolist(), rho_q[1].tolist()
        )
        integrals.append(res)
    integrals = np.array(integrals)

    out = np.empty(81)
    for i in range(81):
        out[i] = limit(all_eps, integrals[:, i], True)
    return out

def test_f(results):
    np.random.seed()
    pt = np.random.rand(3) * 2 - 1.0
    correct = eval(pt)
    for i in range(4):
        interp = barycentric_evalnd(pts, wts, results[:,i], np.array([pt]))[0]
        print("testing:  " + str(i) + "     " + str(
            (correct[i], interp, np.abs((correct[i] - interp) / correct[i]), correct[i] - interp)
        ))

filename = K + 'coincidenttable.npy'

pool = multiprocessing.Pool()
results = np.array(pool.map(eval, pts.tolist()))
np.save(filename, results)
pool.map(test_f, [results for i in range(12)])
