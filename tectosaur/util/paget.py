# The paget algorithm for a = 1
# A quadrature rule for finite-part integrals, D.F. Paget

import numpy as np
from scipy.special import legendre
from tectosaur.util.quadrature import gaussxw, map_to

def calc_bk(N):
    ck = -1.0
    dk = 0.0
    bk = [0.0]
    for k in range(1, N):
        ck = ck * -1.0
        dk = dk + 2.0 / k
        bk.append(ck * dk)
    return bk

def digamma(k):
    s = -np.euler_gamma
    for i in range(1, k):
        s += 1.0 / i
    return s

def calc_bk2(N):
    bk = []
    for k in range(N):
        bk.append(((-1) ** (k + 1)) * (digamma(k + 1) + digamma(k + 1) - 2 * digamma(1)))
    return bk

def calc_e(bk, xs):
    N = len(bk)
    es = []
    for i in range(xs.shape[0]):
        ek = np.zeros(N + 2)
        for k in range(N - 1, -1, -1):
            ek[k] = (
                (k + 0.5) * (bk[k] + (2 * xs[i] * ek[k + 1] / (k + 1)))
                - (((k + 1) * ek[k + 2]) / (k + 2))
            )
        es.append(ek[0])
    return es

def calc_e2(bk, xs):
    N = len(bk)
    es = []
    for i in range(xs.shape[0]):
        s = 0
        for k in range(N):
            s += (k + 0.5) * bk[k] * legendre(k)(xs[i])
        es.append(s)
    return es

def paget(N):
    gx, gw = gaussxw(N)
    bk = calc_bk(N)
    es = calc_e(bk, gx)
    ws = gw * es
    ys = gx / 2 + 0.5
    return ys, ws
