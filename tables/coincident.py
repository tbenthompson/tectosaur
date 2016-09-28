import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

import tectosaur.quadrature as quad

from interpolate import cheb, cheb_wts, barycentric_evalnd, to_interval

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

# def calc(k_name, tri, rho_order, tol, start_eps, eps_step, n_steps, sm, pr):
#     eps = start_eps
#     vals = []
#     rho_gauss = quad.gaussxw(rho_order)
#     for i in range(n_steps):
#         # multiply eps by 2 b/c gauss quad rule is over the interval [-1, 1] whereas
#         # integration interval is [0, 1]
#         rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
#         res = np.array(adaptive_integrate.integrate_coincident(
#             k_name, tri, tol, eps, sm, pr,
#             rho_q[0].tolist(), rho_q[1].tolist()
#         ))
#         vals.append(res)
#         eps /= eps_step
#         # if i > 0:
#         #     print(richardson_limit(eps_step, vals)[0])
#         # else:
#         #     print(vals[0][0])
#     if n_steps > 1:
#         return richardson_limit(eps_step, vals)
#     else:
#         return vals[0]

# These A, B limits come from ablims.py
minlegalA = 0.0939060848748 - 0.01
minlegalB = 0.233648154379 - 0.01
maxlegalB = 0.865565992417 + 0.01

# tol = 0.0001
# rho_order = 100
# eps_start = 0.01
# eps_step = 2.0
# n_steps = 6
# K = "U"

# tol = 1e-4
# rho_order = 100
# eps_start = 1e-6
# eps_step = 2.0
# n_steps = 2
# K = "T"
# K = "A"

# tol = 0.0001
# rho_order = 100
# eps_start = 0.0001
# eps_step = 2.0
# n_steps = 3
# K = "H"

# play parameters
K = "H"
rho_order = 40
starting_eps = 0.1
n_eps = 3
tol = 0.01

n_A = 8
n_B = 8
n_pr = 8

all_eps = starting_eps * 2.0 ** -np.arange(n_eps)
rho_gauss = quad.gaussxw(rho_order)

def AB_val(Ahat, Bhat, prhat):
    A = to_interval(minlegalA, 0.5, Ahat)
    B = to_interval(minlegalB, maxlegalB, Bhat)
    pr = to_interval(0.0, 0.5, prhat)

    tri = [[0,0,0],[1,0,0],[A,B,0.0]]
    print("starting: " + str((A,B)))
    res = calc(
        K, , rho_order, tol, eps_start, eps_step, n_steps, 1.0, pr
    )
    print("finished: " + str((A,B)))
    return res

def calc_entry(pt):
    return AB_val(pt[0], pt[1], pt[2])

def test_interpolation(input):
    np.random.seed()
    idx, vals, pts, wts = input
    Ahat, Bhat, prhat = np.random.rand(3) * 2 - 1
    res = AB_val(Ahat, Bhat, prhat)
    worst = 0.0
    for i in range(81):
        correct = barycentric_evalnd(pts, wts, vals[:,i], np.array([[Ahat,Bhat,prhat]]))[0]
        if correct < 1e-5:
            continue
        est = res[i]
        err = np.abs((correct - est) / correct)
        print(correct, est, err, correct - est)
        worst = max(err, worst)
    return worst

def build_table():
    # n_A = n_B = 8 seems sufficient
    Ahats = cheb(-1, 1, n_A)#minlegalA, 0.5, n_A)
    Bhats = cheb(-1, 1, n_B)#minlegalB, maxlegalB, n_B)
    prhats = cheb(-1, 1, n_pr)
    Ah,Bh,Nh = np.meshgrid(Ahats, Bhats,prhats)
    pts = np.array([Ah.ravel(),Bh.ravel(), Nh.ravel()]).T

    Awts = cheb_wts(-1,1,n_A)
    Bwts = cheb_wts(-1,1,n_B)
    prwts = cheb_wts(-1,1,n_pr)
    wts = np.outer(Awts,np.outer(Bwts,prwts)).ravel()

    load = False
    filename =  str(n_A) + '_' + K + 'k_ABinterptable.npy'
    pool = multiprocessing.Pool()

    if not load:
        vals = []
        vals = pool.map(calc_entry, [pts[i,:] for i in range(pts.shape[0])])
        vals = np.array(vals)
        np.save(filename, vals)
        print(vals.shape)
    else:
        vals = np.load(filename)
        print(vals.shape)
    print("Vals ready")

    n_tests = 12
    results = pool.map(
        test_interpolation,
        zip(
            range(n_tests),
            [vals for i in range(n_tests)],
            [pts for i in range(n_tests)],
            [wts for i in range(n_tests)]
        )
    )
    worst = np.max(results)
    print(worst)

if __name__ == '__main__':
    main()
