import numpy as np
import scipy.integrate
import scipy.linalg
import time

import cppimport
adapt_logic = cppimport.imp('adapt_logic')

def gaussxw(n):
    k = np.arange(1.0, n)
    a_band = np.zeros((2, n))
    a_band[1, 0:n-1] = k / np.sqrt(4 * k * k - 1)
    x, V = scipy.linalg.eig_banded(a_band, lower = True)
    w = 2 * np.real(np.power(V[0,:], 2))
    return x, w

eps = np.finfo(np.float64).eps
pmax = 25
qrs = dict()
for i in range(1, pmax + 1):
    qrs[i] = gaussxw(i)

def map_to(qr, mins, maxs):
    x01s = (qr[0] + 1) / 2
    outx = mins + (maxs - mins) * x01s
    outw = (qr[1] / (2 ** len(mins))) * np.prod(maxs - mins)
    return outx, outw

def quad(f, a, b, q):
    qmapped = map_to(q, (a, b))
    return sum(f(qmapped[0]) * qmapped[1])

def tensor_gauss(nqs):
    d = len(nqs)
    base_rules = [gaussxw(nq) for nq in nqs]
    if d == 1:
        return base_rules[0][0].reshape((nqs[0], 1)), base_rules[0][1]
    mpts = np.meshgrid(*[r[0] for r in base_rules])
    pts = np.array([ax.flatten() for ax in mpts]).T
    wts = np.outer(base_rules[1][1], base_rules[0][1]).flatten()
    for i in range(2, d):
        wts = np.outer(wts, base_rules[i][1]).flatten()
    return pts.reshape(pts.shape, order='C'), wts

def calc_iguess(initial_est, tol, mins, maxs):
    iguess = (tol / eps) * initial_est
    iguess = np.where(iguess < tol, np.prod(maxs - mins), iguess)
    return iguess

# Kahan, Babuska, Neumaier summation
def kbnsum(xs):
    s = xs[0]
    comp = 0
    for x in xs[1:]:
        t = s + x
        if np.abs(s) >= np.abs(x):
            comp += (s - t) + x
        else:
            comp += (x - t) + s
        s = t
    return s, comp

def hadapt_nd_iterative(integrate, mins, maxs, tol, max_refinements = 10000):
    d = len(mins)
    assert(len(maxs) == d)

    initial_cell = getattr(adapt_logic, 'initial_cell' + str(d))
    get_subcell_mins_maxs = getattr(adapt_logic, 'get_subcell_mins_maxs' + str(d))
    refine = getattr(adapt_logic, 'refine' + str(d))

    mins = np.array(mins, dtype = np.float64)
    maxs = np.array(maxs, dtype = np.float64)

    start = time.time()
    int_time = 0

    # TODO: Sometimes, this initial estimate is of a totally different
    # order of magnitude than the final result, meaning that the tolerance
    # is violated. What to do?
    initial_est = integrate(np.array([mins]), np.array([maxs]))[0]
    iguess = calc_iguess(initial_est, tol, mins, maxs)

    cells_left = initial_cell(mins.tolist(), maxs.tolist(), initial_est)
    result = np.zeros_like(iguess)
    count = 1

    for i in range(max_refinements):
        print("cells to compute: " + str(cells_left.size()))

        cell_mins, cell_maxs = get_subcell_mins_maxs(cells_left)

        int_start = time.time()
        cell_integrals = integrate(cell_mins, cell_maxs)
        int_time += time.time() - int_start
        count += len(cell_mins)

        add_to_result, new_cells_left = refine(
            cells_left, cell_mins, cell_maxs, cell_integrals, iguess
        )
        result += add_to_result

        if new_cells_left.size() == 0:
            break
        cells_left = new_cells_left

    print("integrals runtime: " + str(int_time))
    print("logic runtime: " + str(time.time() - start))
    return result, count
