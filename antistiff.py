import numpy as np

import scipy.sparse as sps
from tectosaur.mesh import rect_surface
from tectosaur.quadrature import gauss2d_tri,gauss4d_tri
from tectosaur.util.timer import Timer

pts, tris = rect_surface(30, 30, [[0,0,0],[1,0,0],[1,1,0],[0,1,0]])

NQ = 2
qx, qw = gauss2d_tri(NQ)
nq = qx.shape[0]
nt = tris.shape[0]
print(nt)
n_total_qpts = qx.shape[0] * tris.shape[0]

t = Timer()
tp = pts[tris]

rows = np.tile(
    np.arange(nt * nq * 3).reshape((nt, nq, 3))[:,:,np.newaxis,:], (1,1,3,1)
).flatten()
cols = np.tile(
    np.arange(nt * 9).reshape(nt,3,3)[:,np.newaxis,:,:], (1,nq,1,1)
).flatten()

basis = np.array([1 - qx[:, 0] - qx[:, 1], qx[:, 0], qx[:, 1]]).T
jacobians = np.linalg.norm(
    np.cross(tp[:,2,:] - tp[:,0,:], tp[:,2,:] - tp[:,1,:]),
    axis = 1
)
b_tiled = np.tile((qw[:,np.newaxis] * basis)[np.newaxis,:,:], (nt, 1, 1))
J_tiled = np.tile(jacobians[:,np.newaxis,np.newaxis], (1, nq, 3))
vals = np.tile((J_tiled * b_tiled)[:,:,:,np.newaxis], (1,1,1,3)).flatten()
interp_mat = sps.coo_matrix((vals, (rows, cols)))
galerkin_mat = interp_mat.T
t.report('interp/galerkin')

quad_pts = np.zeros((tris.shape[0] * nq, 3))
for d in range(3):
    for b in range(3):
        quad_pts[:,d] += np.outer(basis[:,b], tp[:,b,d]).T.flatten()

#TODO:
quad_ns = np.array([
    np.zeros(quad_pts.shape[0]),
    np.zeros(quad_pts.shape[0]),
    np.ones(quad_pts.shape[0])
]).T
t.report('quad pts')

np.random.seed(100)
V = np.random.rand(interp_mat.shape[1])
from tectosaur.integral_op import farfield
mat = farfield(1.0, 0.25, pts, tris, tris, NQ)
mat_swapped = np.swapaxes(np.swapaxes(mat, 1, 2), 4, 5)
mat_reshaped = mat_swapped.reshape((nt * 9, nt * 9))
correct = mat_reshaped.dot(V)

from tectosaur.util.gpu import load_gpu
import pycuda.driver as drv
block_size = 1
gpu_module = load_gpu(
    'tectosaur/integrals.cu',
    tmpl_args = dict(block_size = block_size)
)

block = (block_size, 1, 1)
grid = (int(np.ceil(quad_pts.shape[0] / block[0])), 1)
nbody_res = np.zeros((quad_pts.shape[0] * 3)).astype(np.float32)
interp_V = interp_mat.dot(V)
runtime = gpu_module.get_function("farfield_ptsH")(
    drv.Out(nbody_res),
    drv.In(quad_pts.flatten().astype(np.float32)),
    drv.In(quad_ns.flatten().astype(np.float32)),
    drv.In(quad_pts.flatten().astype(np.float32)),
    drv.In(quad_ns.flatten().astype(np.float32)),
    drv.In(interp_V.flatten().astype(np.float32)),
    np.float32(1.0), np.float32(0.25),
    np.int32(quad_pts.shape[0]), np.int32(quad_pts.shape[0]),
    block = block,
    grid = grid,
    time_kernel = True
)

result = galerkin_mat.dot(nbody_res)
np.testing.assert_almost_equal(result, correct, 5)

from tectosaur.integral_op import pairs_quad
far_correction = pairs_quad(1.0, 0.25, pts, tris, tris, gauss4d_tri(NQ), False)
for i in range(tris.shape[0]):
    np.testing.assert_almost_equal(
        mat_swapped[i,:,:,i],
        np.swapaxes(far_correction[i], 1, 2),
        7
    )
