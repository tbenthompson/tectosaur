import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

import tectosaur.quadrature as quad

from interpolate import cheb, cheb_wts, to_interval, barycentric_evalnd
from limit import limit

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

# H parameters
K = "H"
rho_order = 100
starting_eps = 0.0001
n_eps = 4
tol = 0.0001
n_pr = 8
n_theta = 8

# play parameters
# K = "H"
# rho_order = 40
# starting_eps = 0.1
# n_eps = 3
# tol = 0.1
# n_pr = 4
# n_theta = 4

all_eps = starting_eps * 2.0 ** -np.arange(n_eps)
rho_gauss = quad.gaussxw(rho_order)


thetahats = cheb(-1, 1, n_theta)
prhats = cheb(-1, 1, n_pr)
Th,Nh = np.meshgrid(thetahats,prhats)
pts = np.array([Th.ravel(), Nh.ravel()]).T

thetawts = cheb_wts(-1, 1, n_theta)
prwts = cheb_wts(-1, 1, n_pr)
wts = np.outer(thetawts, prwts).ravel()

rho = 0.5 * np.tan(np.deg2rad(20))

def eval(pt):
    thetahat, prhat = pt
    theta = to_interval(0, np.pi, thetahat)
    pr = to_interval(0.0, 0.5, prhat)
    Y = rho * np.cos(theta)
    Z = rho * np.sin(theta)

    tri1 = [[0,0,0],[1,0,0],[0,rho,0]]
    tri2 = [[1,0,0],[0,0,0],[0,Y,Z]]
    integrals = []
    for eps in all_eps:
        print('running: ' + str((pt, eps)))
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        res = adaptive_integrate.integrate_adjacent(
            K, tri1, tri2,
            tol, eps, 1.0, pr, rho_q[0].tolist(), rho_q[1].tolist()
        )
        integrals.append(res)
    integrals = np.array(integrals)

    out = np.empty(81)
    for i in range(81):
        out[i] = limit(all_eps, integrals[:, i], True)
    return out

def test_f(results):
    np.random.seed()
    pt = np.random.rand(2) * 2 - 1.0
    correct = eval(pt)
    for i in range(4):
        interp = barycentric_evalnd(pts, wts, results[:,i], np.array([pt]))[0]
        print("testing:  " + str(i) + "     " + str(
            (correct[i], interp, np.abs((correct[i] - interp) / correct[i]), correct[i] - interp)
        ))

filename = K + 'adjacenttable.npy'

pool = multiprocessing.Pool()
results = np.array(pool.map(eval, pts.tolist()))
np.save(filename, results)
pool.map(test_f, [results for i in range(12)])
