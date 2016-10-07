from tectosaur.dense_integral_op import DenseIntegralOp
import numpy as np

import tectosaur.quadrature as quad

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

A = 0.0
B = 1.0
pr = 0.3

pts = np.array([[0,0,0],[1,0,0],[A,B,0]], dtype = np.float64)
tris = np.array([[0,1,2]])
eps = 0.1 * (2.0 ** -np.arange(1))
op = DenseIntegralOp(eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris)
# op3 = DenseIntegralOp(eps, 16, 15, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris)
op2 = DenseIntegralOp(eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris, use_tables = True)

# rho_order = 50
# rho_gauss = quad.gaussxw(rho_order)
# rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
# res = adaptive_integrate.integrate_coincident(
#     'H', pts[tris][0].tolist(), 0.04, 0.1, 1.0, pr,
#     rho_q[0].tolist(), rho_q[1].tolist()
# )

print(op.mat[0,0])
print(op2.mat[0,0])
# print(res[0])

import ipdb; ipdb.set_trace()
