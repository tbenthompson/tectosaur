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
(DONE) write the table lookup procedure -- multidimensional barycentric lagrange interpolation

TODO:
can i explicitly model and remove the divergence? projects/archive/analytical_bem_integrals/experiments/calc_quad.py has the start of some code to do this based on a modified interpolation basis for the extrapolation process.

test the lookup by selecting random legal triangles and comparing the interpolation
"""

import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

from interpolate import cheb, cheb_wts, barycentric_evalnd, to_interval
import standardize
import tectosaur.quadrature as quad

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
        rho_q = quad.sinh_transform(quad.gaussxw(15), -1, eps)
        res = np.array(adaptive_integrate.integrate(
            k_name, tri, tol, eps, sm, pr, rho_q[0].tolist(), rho_q[1].tolist()
        ))
        vals.append(res)
        eps /= eps_step
        # if i > 0:
        #     print(richardson_limit(eps_step, vals)[0])
        # else:
        #     print(vals[0][0])
    if n_steps > 1:
        return richardson_limit(eps_step, vals)
    else:
        return vals[0]

# These A, B limits come from ablims.py
minlegalA = 0.0939060848748 - 0.01
minlegalB = 0.233648154379 - 0.01
maxlegalB = 0.865565992417 + 0.01

# tol = 0.0001
# eps_start = 0.01
# eps_step = 2.0
# n_steps = 6
# K = "U"

tol = 0.0001
eps_start = 0.0001
eps_step = 2.0
n_steps = 3
K = "H"

def AB_val(Ahat, Bhat, prhat):
    A = to_interval(minlegalA, 0.5, Ahat)
    B = to_interval(minlegalB, maxlegalB, Bhat)
    pr = to_interval(0.0, 0.5, prhat)
    print("starting: " + str((A,B)))
    res = calc(
        K, [[0,0,0],[1,0,0],[A,B,0.0]], tol, eps_start, eps_step, n_steps, 1.0, pr
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

def test_convergence():
    tri = [[0,0,0],[1,0,0],[0.49,0.5,0]]
    # print(adaptive_integrate.integrate("U", tri, 0.01, 0.01, 1.0, 0.25))
    res = calc(
        "U", tri, 0.1, 0.01, 2.0, 6, 1.0, 0.25
    )
    print(res[0])
    # res = calc(
    #     "T", tri, 1e-3, 1e-2, 2.0, 8, 1.0, 0.25
    # )
    # res = calc(
    #     "T", tri, 1e-4, 1e-6, 2.0, 2, 1.0, 0.25
    # )
    # print(res[0])
    # print("NEXT")
    # res = calc(
    #     "T", tri, 1e-4, 1e-7, 2.0, 2, 1.0, 0.25
    # )
    # print(res[0])

def build_test_table():
    # n_A = n_B = 8 seems sufficient
    n_A = 8
    n_B = 8
    n_pr = 8
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

def test_rotation():
    sm = 1.0

    def test_tri(tri):
        standard_tri, labels, R, factor = standardize.standardize(np.array(tri))
        np.testing.assert_almost_equal(
            standard_tri,
            [[0,0,0],[1,0,0],[standard_tri[2][0],standard_tri[2][1],0]]
        )

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
            print("SUCCESS!")
        except Exception as e:
            continue

        n_checks -= 1
        if n_checks == 0:
            break

def main():
    test_convergence()
    # build_test_table()
    # test_rotation()


if __name__ == '__main__':
    main()
