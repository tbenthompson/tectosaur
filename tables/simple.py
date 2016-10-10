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


# rho = 0.5 * np.tan(np.deg2rad(20))
# theta = 2.93114584997056#0.9
# pr = 0.4665063509461097#0.25
#
# pts = np.array([[0,0,0],[1,0,0],[0.5,rho,0.0],[0.5,rho*np.cos(theta),rho*np.sin(theta)]])
# tris = np.array([[0,1,2],[1,0,3]])
# eps = 0.08 * (2.0 ** -np.arange(4))
# op = DenseIntegralOp(eps, 15, 16, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris)
# print(op.mat[0,9])
# print(op.mat[9,0])
#
# from tectosaur.interpolate import barycentric_evalnd, cheb, cheb_wts, from_interval
#
# # from tectosaur.standardize import standardize
# # tri_pts = pts[tris]
# # standard_tri, labels, translation, R, scale = standardize(tri_pts[0,:,:])
# # print(standard_tri, tri_pts)
#
# n_theta, n_pr = 3, 3
#
# thetahats = cheb(-1, 1, n_theta)
# prhats = cheb(-1, 1, n_pr)
# Th,Nh = np.meshgrid(thetahats,prhats)
# interp_pts = np.array([Th.ravel(), Nh.ravel()]).T
#
# thetawts = cheb_wts(-1, 1, n_theta)
# prwts = cheb_wts(-1, 1, n_pr)
# interp_wts = np.outer(thetawts, prwts).ravel()
#
# thetahat = from_interval(0, np.pi, theta)
# prhat = from_interval(0, 0.5, pr)
#
# table = np.load('data/Hadjtesttable.npy')
# print(table[0,0])
# j = 0
# out = barycentric_evalnd(interp_pts, interp_wts, table[:, j], np.array([[thetahat, prhat]]))[0]
# print(out)

# rho_order = 50
# rho_gauss = quad.gaussxw(rho_order)
# rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
# res = adaptive_integrate.integrate_adjacent(
#     'H', pts[tris][0].tolist(), pts[tris][1].tolist(), 0.04, 0.08, 1.0, pr,
#     rho_q[0].tolist(), rho_q[1].tolist()
# )
# print(res[0])
# res = adaptive_integrate.integrate_adjacent(
#     'H', pts[tris][1].tolist(), pts[tris][0].tolist(), 0.04, 0.08, 1.0, pr,
#     rho_q[0].tolist(), rho_q[1].tolist()
# )
# print(res[0])

from tectosaur.standardize import *

obs_tri = np.random.rand(3,3)
fourth_pt = np.random.rand(3)
obs_tri_relabeled, _ = relabel_longest_edge01_and_shortestedge02(obs_tri)
src_tri = np.array([obs_tri_relabeled[1], obs_tri_relabeled[0], fourth_pt])

standard_obs_tri, labels, translation, R, scale = standardize(obs_tri)
standard_src_tri = execute_transformations(src_tri, labels, translation, R, scale)
print(standard_obs_tri, standard_src_tri)
