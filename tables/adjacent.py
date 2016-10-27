import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

import tectosaur.quadrature as quad
import tectosaur.geometry as geometry
from tectosaur.interpolate import cheb, cheb_wts, to_interval, barycentric_evalnd
from tectosaur.limit import limit, richardson_limit

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
K = "H"
rho_order = 50
starting_eps = 0.01
n_eps = 4
tol = 0.01
n_pr = 4
n_theta = 4

filename = (
    '%s_%i_%f_%i_%f_%i_%i_adjacenttable.npy' %
    (K, rho_order, starting_eps, n_eps, tol, n_theta, n_pr)
)
print(filename)

all_eps = starting_eps * 2.0 ** -np.arange(n_eps)
rho_gauss = quad.gaussxw(rho_order)

thetahats = cheb(-1, 1, n_theta)
prhats = cheb(-1, 1, n_pr)
Th,Nh = np.meshgrid(thetahats,prhats)
pts = np.array([Th.ravel(), Nh.ravel()]).T

thetawts = cheb_wts(-1, 1, n_theta)
prwts = cheb_wts(-1, 1, n_pr)
wts = np.outer(thetawts, prwts).ravel()

min_angle = 20
rho = 0.5 * np.tan(np.deg2rad(min_angle))

n_dims = 2

def eval(pt):
    thetahat, prhat = pt
    theta = to_interval(0, np.pi, thetahat)
    pr = to_interval(0.0, 0.5, prhat)
    print(theta, pr)
    Y = rho * np.cos(theta)
    Z = rho * np.sin(theta)

    tri1 = [[0,0,0],[1,0,0],[0.5,rho,0]]
    tri2 = [[1,0,0],[0,0,0],[0.5,Y,Z]]
    integrals = []
    for eps in all_eps:
        print('running: ' + str((pt, eps)))
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        res = adaptive_integrate.integrate_adjacent(
            K, tri1, tri2, tol, eps,
            1.0, pr, rho_q[0].tolist(), rho_q[1].tolist()
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
        if correct[i] < 1e-5:
            continue
        interp = barycentric_evalnd(pts, wts, limits[:,i], np.array([pt]))[0]
        print("testing:  " + str(i) + "     " + str(
            (correct[i], interp, np.abs((correct[i] - interp) / correct[i]), correct[i] - interp)
        ))

pool = multiprocessing.Pool()
results = np.array(pool.map(eval, pts.tolist()))
np.save(filename, results)
pool.map(test_f, zip(range(12), [results for i in range(12)]))
