import attr
import numpy as np
import scipy.sparse

import taskloaf as tsk

import tectosaur.fmm.fmm as fmm
import tectosaur.util.geometry as geometry
import tectosaur.util.gpu as gpu
from tectosaur.farfield import farfield_pts_direct
from tectosaur.util.quadrature import gauss2d_tri, gauss4d_tri
from tectosaur.util.timer import Timer
from tectosaur.kernels import kernels

import logging
logger = logging.getLogger(__name__)

class TriToTriDirectFarfieldOp:
    def __init__(self, nq_far, K_name, params, pts, tris,
            float_type, obs_subset, src_subset):
        self.shape = (obs_subset.shape[0] * 9, src_subset.shape[0] * 9)
        self.dim = pts.shape[1]
        self.tensor_dim = kernels[K_name].tensor_dim
        self.n_obs = obs_subset.shape[0]
        self.n_src = src_subset.shape[0]

        in_size = self.n_src * self.dim * self.tensor_dim
        out_size = self.n_obs * self.dim * self.tensor_dim
        self.gpu_in = gpu.empty_gpu(in_size, float_type)
        self.gpu_out = gpu.empty_gpu(out_size, float_type)

        q = gauss4d_tri(nq_far, nq_far)
        self.gpu_qx = gpu.to_gpu(q[0], float_type)
        self.gpu_qw = gpu.to_gpu(q[1], float_type)
        self.nq = self.gpu_qx.shape[0]

        self.gpu_pts = gpu.to_gpu(pts, float_type)
        self.gpu_obs_tris = gpu.to_gpu(tris[obs_subset], np.int32)
        self.gpu_src_tris = gpu.to_gpu(tris[src_subset], np.int32)
        self.gpu_params = gpu.to_gpu(np.array(params), float_type)
        self.block_size = 128
        self.n_blocks = int(np.ceil(self.n_obs / self.block_size))

        self.module = gpu.load_gpu(
            'farfield_tris.cl',
            tmpl_args = dict(
                block_size = self.block_size,
                float_type = gpu.np_to_c_type(float_type)
            )
        )
        self.fnc = getattr(self.module, "farfield_tris" + K_name)

    async def async_dot(self, tsk_w, v):
        self.gpu_in[:] = v[:].astype(self.gpu_in.dtype)
        self.fnc(
            self.gpu_out, self.gpu_in,
            self.gpu_pts, self.gpu_obs_tris, self.gpu_src_tris,
            self.gpu_qx, self.gpu_qw, self.gpu_params,
            np.int32(self.n_obs), np.int32(self.n_src), np.int32(self.nq),
            grid = (self.n_blocks, 1, 1), block = (self.block_size, 1, 1)
        )
        return self.gpu_out.get()

    def dot(self, v):
        async def wrapper(tsk_w):
            return await self.async_dot(tsk_w, v)
        return tsk.run(wrapper)

    def nearfield_dot(self, v):
        return self.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.dot(v)

    def farfield_dot(self, v):
        return self.dot(v)

class ToPts:
    def __init__(self, nq_far, pts, tris, obs_subset, src_subset):
        far_quad2d = gauss2d_tri(nq_far)
        obs_interp_galerkin_mat, self.obs_quad_pts, self.obs_quad_ns = \
            interp_galerkin_mat(pts[tris[obs_subset]], far_quad2d)
        src_interp_galerkin_mat, self.src_quad_pts, self.src_quad_ns = \
            interp_galerkin_mat(pts[tris[src_subset]], far_quad2d)
        self.obs_galerkin_mat = obs_interp_galerkin_mat.T.tocsr()
        self.src_interp_mat = src_interp_galerkin_mat.tocsr()

    def to_pts(self, v):
        return self.src_interp_mat.dot(v).flatten()

    def from_pts(self, v):
        return self.obs_galerkin_mat.dot(v)

def interp_galerkin_mat(tri_pts, quad_rule):
    nt = tri_pts.shape[0]
    qx, qw = quad_rule
    nq = qx.shape[0]

    rows = np.tile(
        np.arange(nt * nq * 3).reshape((nt, nq, 3))[:,:,np.newaxis,:], (1,1,3,1)
    ).flatten()
    cols = np.tile(
        np.arange(nt * 9).reshape(nt,3,3)[:,np.newaxis,:,:], (1,nq,1,1)
    ).flatten()

    basis = geometry.linear_basis_tri_arr(qx)

    unscaled_normals = geometry.unscaled_normals(tri_pts)
    jacobians = geometry.jacobians(unscaled_normals)

    b_tiled = np.tile((qw[:,np.newaxis] * basis)[np.newaxis,:,:], (nt, 1, 1))
    J_tiled = np.tile(jacobians[:,np.newaxis,np.newaxis], (1, nq, 3))
    vals = np.tile((J_tiled * b_tiled)[:,:,:,np.newaxis], (1,1,1,3)).flatten()

    quad_pts = np.zeros((nt * nq, 3))
    for d in range(3):
        for b in range(3):
            quad_pts[:,d] += np.outer(basis[:,b], tri_pts[:,b,d]).T.flatten()

    scaled_normals = unscaled_normals / jacobians[:,np.newaxis]
    quad_ns = np.tile(scaled_normals[:,np.newaxis,:], (1, nq, 1)).reshape((-1, 3))

    return scipy.sparse.coo_matrix((vals, (rows, cols))), quad_pts, quad_ns


class PtToPtDirectFarfieldOp:
    def __init__(self, nq_far, K_name, params, pts, tris,
            float_type, obs_subset, src_subset):
        self.to_pts = ToPts(nq_far, pts, tris, obs_subset, src_subset)
        self.shape = (obs_subset.shape[0] * 9, src_subset.shape[0] * 9)

        self.params = params
        self.K_name = K_name
        self.float_type = float_type
        self.gpu_obs_pts = gpu.to_gpu(self.to_pts.obs_quad_pts, float_type)
        self.gpu_obs_ns = gpu.to_gpu(self.to_pts.obs_quad_ns, float_type)
        self.gpu_src_pts = gpu.to_gpu(self.to_pts.src_quad_pts, float_type)
        self.gpu_src_ns = gpu.to_gpu(self.to_pts.src_quad_ns, float_type)

    async def async_dot(self, tsk_w, v):
        interp_v = self.to_pts.to_pts(v)
        nbody_result = farfield_pts_direct(
            self.K_name, self.gpu_obs_pts, self.gpu_obs_ns,
            self.gpu_src_pts, self.gpu_src_ns, interp_v, self.params, self.float_type
        )
        out = self.to_pts.from_pts(nbody_result)
        return out

    def dot(self, v):
        async def wrapper(tsk_w):
            return await self.async_dot(tsk_w, v)
        return tsk.run(wrapper)

    def nearfield_dot(self, v):
        return self.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.dot(v)

    def farfield_dot(self, v):
        return self.dot(v)

@attr.s()
class PtToPtFMMFarfieldOp:
    order = attr.ib()
    mac = attr.ib()
    pts_per_cell = attr.ib()
    def __call__(self, nq_far, K_name, params, pts, tris, float_type,
            obs_subset, src_subset):
        return PtToPtFMMFarfieldOpImpl(
            nq_far, K_name, params, pts, tris, float_type,
            obs_subset, src_subset, self.order, self.mac, self.pts_per_cell
        )

class PtToPtFMMFarfieldOpImpl:
    def __init__(self, nq_far, K_name, params, pts, tris, float_type,
            obs_subset, src_subset, order, mac, pts_per_cell):

        self.to_pts = ToPts(nq_far, pts, tris, obs_subset, src_subset)
        obs_pts = self.to_pts.obs_quad_pts
        obs_ns = self.to_pts.obs_quad_ns
        src_pts = self.to_pts.src_quad_pts
        src_ns = self.to_pts.src_quad_ns

        cfg = fmm.make_config(K_name, params, 1.1, mac, order, float_type)

        self.obs_tree = fmm.make_tree(obs_pts.copy(), cfg, pts_per_cell)
        self.src_tree = fmm.make_tree(src_pts.copy(), cfg, pts_per_cell)
        self.obs_orig_idxs = np.array(self.obs_tree.orig_idxs)
        self.src_orig_idxs = np.array(self.src_tree.orig_idxs)
        self.fmm_obj = fmm.FMM(
            self.obs_tree, obs_ns[self.obs_orig_idxs].copy(),
            self.src_tree, src_ns[self.src_orig_idxs].copy(), cfg
        )
        fmm.report_interactions(self.fmm_obj)
        self.evaluator = fmm.FMMEvaluator(self.fmm_obj)

    async def async_dot(self, tsk_w, v):
        t = Timer(output_fnc = logger.debug)
        pts_v = self.to_pts.to_pts(v)
        input_tree = pts_v.reshape((-1,3))[self.src_orig_idxs,:].reshape(-1)
        t.report('to tree space')

        fmm_out = await self.evaluator.eval(tsk_w, input_tree.copy())
        fmm_out = fmm_out.reshape((-1, 3))
        t.report('fmm eval')

        to_orig = np.empty_like(fmm_out)
        to_orig[self.obs_orig_idxs,:] = fmm_out
        to_orig = to_orig.flatten()
        from_pts = self.to_pts.from_pts(to_orig)
        t.report('to output space')
        return from_pts

    def dot(self, v):
        async def wrapper(tsk_w):
            return await self.async_dot(tsk_w, v)
        return tsk.run(wrapper)

    def nearfield_dot(self, v):
        return self.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.dot(v)

    def farfield_dot(self, v):
        return self.dot(v)
