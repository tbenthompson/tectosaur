from tectosaur.dense_integral_op import DenseIntegralOp
import numpy as np

import tectosaur.quadrature as quad

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

# A = 0.0
# B = 1.0
# pr = 0.3
#
# pts = np.array([[0,0,0],[1,0,0],[A,B,0]], dtype = np.float64)
# tris = np.array([[0,1,2]])
# eps = 0.1 * (2.0 ** -np.arange(1))
# op = DenseIntegralOp(eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris)
# # op3 = DenseIntegralOp(eps, 16, 15, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris)
# op2 = DenseIntegralOp(eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris, use_tables = True)
#
# # rho_order = 50
# # rho_gauss = quad.gaussxw(rho_order)
# # rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
# # res = adaptive_integrate.integrate_coincident(
# #     'H', pts[tris][0].tolist(), 0.04, 0.1, 1.0, pr,
# #     rho_q[0].tolist(), rho_q[1].tolist()
# # )
#
# print(op.mat[0,0])
# print(op2.mat[0,0])
# print(res[0])


rho = 0.5 * np.tan(np.deg2rad(20))
theta = 0.2
pr = 0.25

pts = np.array([[0,0,0],[1,0,0],[0.5,rho,0.0],[0,rho*np.cos(theta),rho*np.sin(theta)]])
tris = np.array([[0,1,2],[1,0,3]])
eps = 0.08 * (2.0 ** -np.arange(1))
op = DenseIntegralOp(eps, 15, 20, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris)
print(op.mat[0,9])
print(op.mat[9,0])

# rho_order = 50
# rho_gauss = quad.gaussxw(rho_order)
# rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
# res = adaptive_integrate.integrate_adjacent(
#     'H', pts[tris][0].tolist(), pts[tris][1].tolist(), 0.004, 0.01, 1.0, pr,
#     rho_q[0].tolist(), rho_q[1].tolist()
# )
# print(res[0])
# res = adaptive_integrate.integrate_adjacent(
#     'H', pts[tris][1].tolist(), pts[tris][0].tolist(), 0.004, 0.01, 1.0, pr,
#     rho_q[0].tolist(), rho_q[1].tolist()
# )
# print(res[0])


import ipdb; ipdb.set_trace()
