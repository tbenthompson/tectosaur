import sys
import numpy as np
import scipy.integrate as spi
import scipy.interpolate
import matplotlib.pyplot as plt
import cppimport
ext = cppimport.imp('ext')

# ls = 2.0 ** np.arange(-2,3)
# eps = 2.0 ** -np.arange(0,10)
# print(eps)
#
# nu = 0.25
# tracC1 = 1.0/(4*np.pi*(1-nu))
# tracC2 = 1.0-2*nu
# kronecker = [[1,0,0],[0,1,0],[0,0,1]]
# k = 1
# j = 1
# def f(obsp, obsn, srcp, srcn, eps):
#     obsoffset = obsp + eps * obsn
#     delta = srcp - obsoffset
#     r2 = np.sum(delta ** 2)
#     r = np.sqrt(r2)
#     drdn = np.sum(delta * srcn) / r
#     return -(tracC1 / r) * (
#         (tracC2 * kronecker[k][j] + 2 * delta[j] * delta[k] / r2) * drdn
#         - tracC2 * (srcn[j] * delta[k] - srcn[k] * delta[j]) / r
#     )
#     # return np.sum(delta * srcn) / (2 * r2);
#
# e = eps[6]
# for L in ls:
#     # for e in eps:
#     releps = L * e
#     print(spi.quad(lambda x: spi.quad(
#         lambda y: f(
#             np.array([x, 0]),
#             np.array([0, 1]),
#             np.array([y, 0]),
#             np.array([0, 1]),
#             releps
#         ), -L, L
#     )[0], -L, L))

for p in range(2, 20):
    nus_t = np.linspace(0.0, 0.5, p)
    nus_e = np.linspace(0.0, 0.5, 500)
    vals_t = np.array([ext.evalU(0,0,0,0,0,1,1,0,0,0,0,1,1.0,nu)[-1] for nu in nus_t])
    vals_e = np.array([ext.evalU(0,0,0,0,0,1,1,0,0,0,0,1,1.0,nu)[-1] for nu in nus_e])
    vals_e_test = scipy.interpolate.barycentric_interpolate(nus_t, vals_t, nus_e)
    err = np.abs((vals_e_test - vals_e) / vals_e)
    print(p, np.max(err))
import ipdb; ipdb.set_trace()

# for v in :
#     print()
#     print(ext.evalT(0,0,0,0,0,1,1,0,0,0,0,1,1.0,v))
#     print(ext.evalH(0,0,0,0,0,1,1,0,0,0,0,1,1.0,v))
# sys.exit()


eps = 0.5
L = 1.0
A = 1.0
B = 1.0
def get_pt(xhat, yhat):
    return xhat * L + yhat * A * L, yhat * B * L, 0

theta_lims = [
    lambda x, y: np.pi - np.arctan((1 - y) / x),
    lambda x, y: np.pi + np.arctan(y / x),
    lambda x, y: 2 * np.pi - np.arctan(y / (1 - x))
]
rho_lims = [
    lambda x, y, t: (1 - y - x) / (np.cos(t) + np.sin(t)),
    lambda x, y, t: -x / np.cos(t),
    lambda x, y, t: -y / np.sin(t)
]

npts = 0
OXX = []
OYY = []
SXX = []
SYY = []
for i in range(9):
    def f(k, x, y, xhat, yhat, eps):
        global npts
        npts+=1
        if npts % 10000000 == 0:
            print(npts)
        #     plt.figure()
        #     plt.plot(SXX, SYY, '.')
            plt.figure()
            plt.plot(OXX, OYY, '.')
            plt.show()
        if k == 0:
            theta = (1 - xhat) * theta_lims[0](x, y) + xhat * theta_lims[1](x, y)
            rho = yhat * rho_lims[1](x, y, theta)
        elif k == 1:
            theta = (1 - xhat) * theta_lims[1](x, y) + xhat * theta_lims[2](x, y)
            rho = yhat * rho_lims[2](x, y, theta)
        elif k == 2:
            theta = (1 - xhat) * (theta_lims[2](x, y) - 2 * np.pi) + xhat * theta_lims[0](x, y)
            rho = yhat * rho_lims[0](x, y, theta)
        ox, oy, oz = get_pt(x, y)
        oz += eps
        xstar = x + rho * np.cos(theta)
        ystar = y + rho * np.sin(theta)
        OXX.append(x);OYY.append(y);#SXX.append(theta);SYY.append(rho)
        sx, sy, sz = get_pt(xstar, ystar)
        return rho * ext.evalU(
            ox, oy, oz,
            0, 0, 1,
            sx, sy, sz,
            0, 0, 1,
            1.0, 0.25
        )[i]

    I = 0
    for k in range(3):
        I += spi.quad(
            lambda x: spi.quad(
                lambda y: spi.quad(
                    lambda xstar: spi.quad(
                        lambda ystar: f(k, x, y, xstar, ystar, eps),
                        0, 1
                    )[0],
                    0, 1
                )[0],
                0, 1 - x
            )[0],
            0, 1
        )[0]
    print(I)


