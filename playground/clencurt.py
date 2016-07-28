from scipy.fftpack import ifft
import numpy as np
from tectosaur.quadrature import *
from tectosaur.elastic import *
import sympy

kernel = H(0, 0)
kernel_lambda = sympy.utilities.lambdify(all_args, kernel['expr'], "numpy")
params = np.random.rand(16)
x = [np.zeros(10),np.zeros(10),np.zeros(10)]
y = [np.zeros(10) + 1,np.zeros(10),np.zeros(10)]
l = [0, 0, 1]
n = [0, 0, 1]
params = [1.0, 0.25, 0, 0] + x + y + l + n
result = kernel_lambda(*params)

def clencurt(n1):
    """ Computes the Clenshaw Curtis nodes and weights """
    if n1 == 1:
        x = 0
        w = 2
    else:
        n = n1 - 1
        C = np.zeros((n1,2))
        k = 2*(1+np.arange(np.floor(n/2)))
        C[::2,0] = 2/np.hstack((1, 1-k*k))
        C[1,1] = -n
        V = np.vstack((C,np.flipud(C[1:n,:])))
        F = np.real(ifft(V, n=None, axis=0))
        x = F[0:n1,1]
        w = np.hstack((F[0,0],2*F[1:n,0],F[n,0]))
    return x,w
#
# for i in range(3, 10):
#     q = clencurt(i)
#     result = quadrature(lambda x: np.sin(x) ** 2, q)
#     print(result)

# for i in range(3, 1000, 5):
#     print(i, quadrature(f, clencurt(i)) - 19999)
#
# q5 = clencurt(5)
# qsides = clencurt(20)
# for n_splits in range(2, 100):
#     bounds = np.linspace(-0.1, 0.1, n_splits)
#     result = quadrature(f, map_to(qsides, (-1, -0.1)))
#     result += quadrature(f, map_to(qsides, (0.1, 1.0)))
#     for i in range(n_splits - 1):
#         start = bounds[i]
#         end = bounds[i + 1]
#         q_mapped = map_to(q5, (start, end))
#         result += quadrature(f, q_mapped)
#     print(n_splits * 4 + 2 + qsides[0].shape[0] * 2, result - 19999)

eps = 0.01
correct = 2.0/(eps**2 * np.sqrt(1.0+eps**2.0))
correct = 2.0 * np.arctan(1.0 / eps) / eps
PTS = 0
def f(x):
    global PTS
    PTS += 1
    return 1.0 / ((x ** 2 + eps ** 2) ** 1.0)

qs = [clencurt(nq) for nq in [3, 5, 9, 17, 33, 65, 129]]
result = 0
print(qs[0][0])

def map_pts(X, a, b):
    return a + (X / 2 + 0.5) * (b - a)

def integrate(fnc, a, b, p):
    return sum(np.array([fnc(xv) for xv in map_pts(qs[p][0], a, b)]) * (b - a) * qs[p][1] / 2.0)

def aqcc(fnc, tol, a, b, p, p_refine = False):
    R1 = integrate(fnc, a, b, p)
    print(a,b,R1,p,p_refine)
    R2 = integrate(fnc, a, b, p + 1)
    err = np.abs((R2 - R1) / R2)

    if err < tol:
        return R2

    qx = qs[p + 1][0]
    fs = np.array([fnc(xv) for xv in qx])
    deltaX = qx[1:] - qx[:-1]
    deltaF = fs[1:] - fs[:-1]
    deriv_mag = np.abs(deltaF / deltaX)
    should_h_adapt = np.logical_or(deriv_mag > np.median(deriv_mag) * 100, p >= len(qs) - 2)
    h_adapt_region = should_h_adapt.nonzero()[0]
    p_adapt_regions = [(0, qx.shape[0] - 2)]
    for h_r in h_adapt_region:
        new_p_adapt_regions = []
        for p_r in p_adapt_regions:
            if h_r >= p_r[0] and h_r <= p_r[1]:
                if h_r - 1 - p_r[0] > 0:
                    new_p_adapt_regions.append((p_r[0], h_r - 1))
                if p_r[1] - h_r - 1:
                    new_p_adapt_regions.append((h_r + 1, p_r[1]))
            else:
                new_p_adapt_regions.append(p_r)
        p_adapt_regions = new_p_adapt_regions

    result = 0
    for h_r in h_adapt_region:
        L = map_pts(qx[h_r], a, b)
        R = map_pts(qx[h_r + 1], a, b)
        result += aqcc(fnc, tol, L, R, 0) # could also recurse with order p
    for p_r in p_adapt_regions:
        L = map_pts(qx[p_r[0]], a, b)
        R = map_pts(qx[p_r[1] + 1], a, b)
        if len(h_adapt_region) > 0:
            result += aqcc(fnc, tol, L, R, p, True)
        else:
            result += aqcc(fnc, tol, L, R, p + 1, True)
    return result

print((aqcc(f, 1e-1, -1, 1, 1) - correct) / correct)
PTS = 0
print((scipy.integrate.quad(f, -1, 1, epsabs = 1e-1)[0] - correct) / correct)
print(PTS)
# for play in 2 ** np.linspace(np.log(eps) / np.log(2) - 5, np.log(eps) / np.log(2) + 5):
#     print(play,(quadrature(f, sinh_transform(gaussxw(20), 0, play)) - correct) / correct)


PTS=0
def fdbl(x, y):
    global PTS
    PTS+=1
    return 1.0 / ((x - y) ** 2 + eps ** 2) ** 1.5
res = aqcc(lambda x: aqcc(lambda y: fdbl(x, y), 1e-1, -1, 1, 0), 1e-1, -1, 1, 0)
# res = scipy.integrate.dblquad(fdbl, -1, 1, lambda x: -1, lambda x: 1, epsabs = 1e-2)
print(res)
print(PTS)


# try adaptive 3-5 triangular quadrature rule - simpson/trapezoidal? gauss? chebyshev?
# hp adaptive quadrature? detect whether there are large derivatives in the region and decide whether to increase order or increase point density.
# try taylor series method -- the problem is that it doesn't work well near element endpoints. how much of a problem is that? a lot!
