# The paget algorithm for a = 1
# A quadrature rule for finite-part integrals, D.F. Paget

import numpy as np
from math import factorial
from scipy.special import legendre, gamma
from tectosaur.util.quadrature import gaussxw, map_to

def calc_bk(N, a):
    bk = [-1] * (a - 1)
    if a == 1:
        ck = -1.0
    else:
        ck = (2 * a - 2) / (1 - a)
    dk = 0.0
    for r in range(1, a):
        dk += 1.0 / (r * (r + a - 1))
    dk *= -(a - 1)
    bk.append(ck * dk)
    for k in range(a, N):
        ck = -ck * (k + a - 1) / (k - a + 1)
        dk = dk + (1.0 / (k + a - 1)) + (1.0 / (k - a + 1))
        bk.append(ck * dk)
    return bk

def digamma(k):
    s = -np.euler_gamma
    for i in range(1, k):
        s += 1.0 / i
    return s

def gammaratio(a, k):
    return gamma(a + k) / gamma(a)

def calc_bk2(N, a):
    bk = []
    for k in range(N):
        if k < a - 1:
            bk.append(-1)
            continue
        bk.append(
            ((-1) ** (k + a) * factorial(k + a - 1) / (factorial(k - a + 1) * factorial(a - 1)))
            * (digamma(k + a) + digamma(k - a + 2) - 2 * digamma(a))
        )
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

# Note: this returns a quadrature rule for a finite part integral with the
# singularity at x = -1, integrate(f(x) / (x + 1), x, -1, 1)
def paget(N, a):
    gx, gw = gaussxw(N)
    bk = calc_bk(N, a)
    es = calc_e(bk, gx)
    ws = gw * es * 2
    return gx, ws
