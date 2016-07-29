from tectosaur.quadrature import *
from tectosaur.geometry import *
from tectosaur.elastic import *
import matplotlib.pyplot as plt
import sympy
import autograd.numpy as np
from autograd import grad

def hyp(x, y, n, N):
    n = np.array(n)
    N = np.array(N)
    rvec = x - y
    r2 = sum(rvec ** 2)
    ndN = sum(n * N)
    ndR = sum(n * rvec)
    NdR = sum(rvec * N)
    return (ndN - (2 * ndR * NdR / r2)) / (2 * np.pi * r2)

# res = scipy.integrate.quadrature(lambda x: hyp(np.array([0, 1.0]), [x, 0], [0, 1], [0, 1]), -1, 1)
# print(res)
#
#
# x = [0, 0.5]
# n = 100
# ys = np.array([np.linspace(-1, 1, n), np.zeros(n)]).T
#
# res = np.empty((4, n))
# for i in range(n):
#     hyp_grad = grad(lambda xy: hyp(np.array([0, xy]), ys[i,:], [0, 1], [0, 1]))
#     hyp_grad2 = grad(lambda xy: hyp_grad(xy))
#     hyp_grad3 = grad(lambda xy: hyp_grad2(xy))
#     res[0, i] = hyp(x, ys[i,:], [0, 1], [0, 1])
#     res[1, i] = hyp_grad(x[1])
#     res[2, i] = hyp_grad2(x[1])
#     res[3, i] = hyp_grad3(x[1])
# for i in range(res.shape[0]):
#     plt.plot(ys[:,0],res[i,:] / np.mean(res[i, :]))
# plt.savefig('abc.pdf')

# for xy in np.linspace(-1, 1, 100):
#     print(f(0.3))

def taylor_series(f, n):
    if n == 0:
        return [f]
    return [f] + taylor_series(grad(f), n - 1)

q = gaussxw(2000)
def eval_taylor(ts, x, y):
    global q
    result = 0
    for i in range(len(ts)):
        q = gaussxw(18)
        te1 = ((y - x) ** i / factorial(i)) * ts[i](x)
        q = gaussxw(20)
        te2 = ((y - x) ** i / factorial(i)) * ts[i](x)
        print(np.abs((te1 - te2)), te2)
        result += te2
    return result

f_outer = lambda outer_x: lambda dom: lambda y: quadrature(lambda x: hyp([outer_x, y], np.array([x - dom, 0]), [0, 1], [0, 1]), q)
def calc_inner(x):
    f_dom = f_outer(x)
    hs = 0.5 * 2.0 ** -np.arange(7)
    fs = [f_dom(0)(h) + f_dom(2)(h) for h in hs]
    res = richardson(np.array(hs), np.array(fs))
    return res[-1]
print(calc_inner(-1.0))
print(scipy.integrate.quad(calc_inner, -2, 0))

q = gaussxw(30)
# q = sinh_transform(gaussxw(50), 0, 0.5)
# print('std::vector<double> qx = {' + ','.join(map(str, q[0])) + '};')
# print('std::vector<double> qw = {' + ','.join(map(str, q[1])) + '};')
# import ipdb; ipdb.set_trace()
def calc_inner_taylor(x):
    print(x)
    f_dom = f_outer(x)
    return sum([eval_taylor(taylor_series(f_dom(x), 7), 1.0, 0.0) for x in [0, 2, -2]])

print(calc_inner_taylor(0.0))

# q5 = gaussxw(4)
# shiftedqx = q5[0] - 1
# res = quadrature(lambda xs: np.array([calc_inner_taylor(x) for x in xs]), (shiftedqx, q5[1]))
# print(res)

# xs = np.linspace(-3, 3, 100)
# vs = [calc_inner_taylor(x) for x in xs]
# plt.plot(xs, vs)
# plt.savefig('abc.pdf')

# print(scipy.integrate.quad(calc_inner_taylor, -2, 0))


# find the nearfield expansion points, these need to be far from any of the adjacent elements
# calculate the source integrals for coincident/edgeadj/vertadj for each basis pair interaction and for each observation quadrature point --> this should do taylor series automatic differentiation for coincident and edgeadj
# extrapolate to 0 using the taylor series for coincident and edge adj
# perform observation quadrature
# nearfield!
