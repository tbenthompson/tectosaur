import multiprocessing
import matplotlib.pyplot as plt
import numpy as np

import tectosaur.quadrature as quad
from tectosaur.geometry import tri_normal

import interpolate
from remove_divergence import make_terms

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

def remove_proj(V, b):
    return V - (V.dot(b) * b) / np.linalg.norm(b)

def vec_angle(v1, v2):
    return np.arccos(v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2))

def get_offset(tri1, tri2):
    p = tri1[1] - tri1[0]
    L1 = tri1[2] - tri1[0]
    L2 = tri2[2] - tri2[0]
    T1 = remove_proj(L1, p)
    T2 = remove_proj(L2, p)
    n1 = tri_normal(tri1, normalize = True)

    theta = vec_angle(T1, T2)
    samedir = n1.dot(T2 - T1) > 0
    if not samedir:
        theta = 2 * np.pi - theta

    v = n1
    if theta > np.pi:
        v = -(T1 + T2) / 2.0
        v /= np.linalg.norm(v)
    return v

def plot_offset(Y, Z, offset, eps = 0.05):
    plt.plot([0, Y], [0, Z])
    plt.plot([0, 1], [0, 0])
    plt.plot([-offset[1] * eps, 1 - offset[1] * eps], [-offset[2] * eps, -offset[2] * eps])
    plt.xlim([-1.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.show()


K = "H"
# rho_order = 80
# starting_eps = 0.0001
# n_eps = 4
# tol = 0.0001
# n_pr = 8
# n_theta = 8

rho_order = 40
starting_eps = 0.1
n_eps = 3
tol = 0.01
n_pr = 4
n_theta = 4

all_eps = starting_eps * 2.0 ** -np.arange(n_eps)
rho_gauss = quad.gaussxw(rho_order)


thetahats = interpolate.cheb(-1, 1, n_theta)
prhats = interpolate.cheb(-1, 1, n_pr)
Th,Nh = np.meshgrid(thetahats,prhats)
pts = np.array([Th.ravel(), Nh.ravel()]).T

thetawts = interpolate.cheb_wts(-1, 1, n_theta)
prwts = interpolate.cheb_wts(-1, 1, n_pr)
wts = np.outer(thetawts, prwts).ravel()

rho = 0.5 * np.tan(np.deg2rad(20))

def eval(thetahat, prhat):
    theta = interpolate.to_interval(0, np.pi, thetahat)
    pr = interpolate.to_interval(0.0, 0.5, prhat)
    Y = rho * np.cos(theta)
    Z = rho * np.sin(theta)

    tri1 = [[0,0,0],[1,0,0],[0,rho,0]]
    tri2 = [[1,0,0],[0,0,0],[0,Y,Z]]
    integrals = []
    for eps in all_eps:
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        res = adaptive_integrate.integrate_adjacent(
            K, tri1, tri2, [0, 0, 1],
            tol, eps, 1.0, pr, rho_q[0].tolist(), rho_q[1].tolist()
        )
        integrals.append(res)
    integrals = np.array(integrals)

    terms = make_terms(len(all_eps), True)
    mat = [[t(e) for t in terms] for e in all_eps]
    coeffs = np.linalg.solve(mat, integrals[:, 0])

    return coeffs[1]

def parallel_shim(pt):
    return eval(pt[0], pt[1])

pool = multiprocessing.Pool()
results = pool.map(parallel_shim, pts.tolist())

np.save('adjacenttable.npy', results)

# interp_theta = (interp_thetahat + 1.0) * np.pi / 2.0
# plt.plot(interp_theta, results)
# plt.show()

for i in range(10):
    thetahat, prhat = np.random.rand(2) * 2 - 1.0
    correct = eval(thetahat, prhat)
    interp = interpolate.barycentric_evalnd(pts, wts, results, np.array([[thetahat, prhat]]))[0]
    print("testing: " + str((correct, interp, np.abs((correct - interp) / correct), correct - interp)))

    # from tectosaur.dense_integral_op import DenseIntegralOp
    # op = DenseIntegralOp(
    #     [eps], 5, 20, 5, 5, 5, 5, K, 1.0, 0.25,
    #     np.vstack((tri1, tri2)), np.array([[0,1,2],[1,0,5]])
    # )
    # print(op.mat[0,9])
