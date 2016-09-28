import numpy as np
import scipy.integrate as spi


ls = 2.0 ** np.arange(-2,3)
eps = 2.0 ** -np.arange(0,10)
print(eps)

nu = 0.25
tracC1 = 1.0/(4*np.pi*(1-nu))
tracC2 = 1.0-2*nu
kronecker = [[1,0,0],[0,1,0],[0,0,1]]
k = 1
j = 1
def f(obsp, obsn, srcp, srcn, eps):
    obsoffset = obsp + eps * obsn
    delta = srcp - obsoffset
    r2 = np.sum(delta ** 2)
    r = np.sqrt(r2)
    drdn = np.sum(delta * srcn) / r
    return -(tracC1 / r) * (
        (tracC2 * kronecker[k][j] + 2 * delta[j] * delta[k] / r2) * drdn
        - tracC2 * (srcn[j] * delta[k] - srcn[k] * delta[j]) / r
    )
    # return np.sum(delta * srcn) / (2 * r2);

e = eps[6]
for L in ls:
    # for e in eps:
    releps = L * e
    print(spi.quad(lambda x: spi.quad(
        lambda y: f(
            np.array([x, 0]),
            np.array([0, 1]),
            np.array([y, 0]),
            np.array([0, 1]),
            releps
        ), -L, L
    )[0], -L, L))










# import cppimport
# ext = cppimport.imp('ext')
#
# print(ext.evalU(0,0,0,0,0,1,1,0,0,0,0,1,1.0,0.25))
