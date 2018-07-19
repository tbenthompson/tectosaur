import os
import attr
import numpy as np
import scipy.sparse

import taskloaf as tsk

from tectosaur.farfield import farfield_pts_direct

import tectosaur.fmm.fmm as fmm
from tectosaur.nearfield.nearfield_op import NearfieldIntegralOp
from tectosaur.nearfield.table_lookup import coincident_table, adjacent_table

from tectosaur.util.quadrature import gauss2d_tri
import tectosaur.util.geometry as geometry
import tectosaur.util.gpu as gpu
from tectosaur.util.timer import Timer

import logging
logger = logging.getLogger(__name__)

# Handy wrapper for creating an integral op
def make_integral_op(pts, tris, k_name, k_params, cfg, obs_subset, src_subset):
    if cfg['use_fmm']:
        farfield = FMMFarfieldBuilder(
            cfg['fmm_order'], cfg['fmm_mac'], cfg['pts_per_cell']
        )
    else:
        farfield = None
    return SparseIntegralOp(
        cfg['quad_vertadj_order'], cfg['quad_far_order'],
        cfg['quad_near_order'], cfg['quad_near_threshold'],
        k_name, k_params, pts, tris, cfg['float_type'],
        farfield_op_type = farfield,
        obs_subset = obs_subset,
        src_subset = src_subset
    )

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

class DirectFarfield:
    def __init__(self, kernel, params, obs_pts, obs_ns, src_pts, src_ns, float_type):
        self.params = params
        self.kernel = kernel
        self.float_type = float_type
        self.gpu_obs_pts = gpu.to_gpu(obs_pts, float_type)
        self.gpu_obs_ns = gpu.to_gpu(obs_ns, float_type)

        if src_pts is obs_pts:
            self.gpu_src_pts = self.gpu_obs_pts
        else:
            self.gpu_src_pts = gpu.to_gpu(src_pts, float_type)

        if src_ns is obs_ns:
            self.gpu_src_ns = self.gpu_obs_ns
        else:
            self.gpu_src_ns = gpu.to_gpu(src_ns, float_type)

    async def async_dot(self, tsk_w, v):
        return farfield_pts_direct(
            self.kernel, self.gpu_obs_pts, self.gpu_obs_ns,
            self.gpu_src_pts, self.gpu_src_ns, v, self.params, self.float_type
        )

class FMMFarfield:
    def __init__(self, kernel, params, obs_pts, obs_ns, src_pts, src_ns,
            float_type, order, mac, pts_per_cell):
        cfg = fmm.make_config(kernel, params, 1.1, mac, order, float_type)

        # TODO: different obs and src pts
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
        input_tree = v.reshape((-1,3))[self.src_orig_idxs,:].reshape(-1)
        t.report('to tree space')

        fmm_out = await self.evaluator.eval(tsk_w, input_tree.copy())
        fmm_out = fmm_out.reshape((-1, 3))
        t.report('fmm eval')

        to_orig = np.empty_like(fmm_out)
        to_orig[self.obs_orig_idxs,:] = fmm_out
        to_orig = to_orig.flatten()
        t.report('to output space')
        return to_orig

    def dot(self, v):
        async def wrapper(tsk_w):
            return await self.async_dot(tsk_w, v)
        return tsk.run(wrapper)

@attr.s()
class FMMFarfieldBuilder:
    order = attr.ib()
    mac = attr.ib()
    pts_per_cell = attr.ib()

    def __call__(self, kernel, params, obs_pts, obs_ns, src_pts, src_ns, float_type):
        return FMMFarfield(
            kernel, params, obs_pts, obs_ns, src_pts, src_ns, float_type,
            self.order, self.mac, self.pts_per_cell
        )

class FarfieldOp:
    def __init__(self, nq_far, kernel, params, pts, tris, float_type,
            farfield_op_type = None, obs_subset = None, src_subset = None):
        far_quad2d = gauss2d_tri(nq_far)
        self.obs_interp_galerkin_mat, obs_quad_pts, obs_quad_ns = \
            interp_galerkin_mat(pts[tris[obs_subset]], far_quad2d)
        self.src_interp_galerkin_mat, src_quad_pts, src_quad_ns = \
            interp_galerkin_mat(pts[tris[src_subset]], far_quad2d)

        self.obs_interp_galerkin_mat = self.obs_interp_galerkin_mat.tocsr()
        self.src_interp_galerkin_mat = self.src_interp_galerkin_mat.tocsr()

        if farfield_op_type is None:
            farfield_op_type = DirectFarfield

        self.farfield_op = farfield_op_type(
            kernel, params, obs_quad_pts, obs_quad_ns, src_quad_pts, src_quad_ns, float_type
        )
        self.shape = (obs_subset.shape[0] * 9, src_subset.shape[0] * 9)

    async def async_dot(self, tsk_w, v):
        t = Timer(output_fnc = logger.debug)
        logger.debug("start farfield_dot")
        interp_v = self.src_interp_galerkin_mat.dot(v).flatten()
        nbody_result = await self.farfield_op.async_dot(tsk_w, interp_v)
        #TODO: pre-transpose
        out = self.obs_interp_galerkin_mat.T.dot(nbody_result)
        t.report('farfield_dot')
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

class SparseIntegralOp:
    def __init__(self, nq_vert_adjacent, nq_far, nq_near, near_threshold,
            kernel, params, pts, tris, float_type, farfield_op_type = None,
            obs_subset = None, src_subset = None):

        if obs_subset is None:
            obs_subset = np.arange(tris.shape[0])
        if src_subset is None:
            src_subset = np.arange(tris.shape[0])

        self.nearfield = NearfieldIntegralOp(
            pts, tris, obs_subset, src_subset,
            nq_vert_adjacent, nq_far, nq_near,
            near_threshold, kernel, params, float_type
        )

        self.farfield = FarfieldOp(
            nq_far, kernel, params, pts, tris, float_type, farfield_op_type,
            obs_subset = obs_subset, src_subset = src_subset
        )

        self.shape = self.nearfield.shape
        self.gpu_nearfield = None

    async def nearfield_dot(self, v):
        t = Timer(output_fnc = logger.debug)
        logger.debug("start nearfield_dot")
        out = self.nearfield.dot(v)
        t.report("nearfield_dot")
        return out

    async def nearfield_no_correction_dot(self, v):
        return self.nearfield.nearfield_no_correction_dot(v)

    async def async_dot(self, tsk_w, v):
        async def call_farfield(tsk_w):
            return (await self.farfield_dot(tsk_w, v))
        async def call_nearfield(tsk_w):
            return (await self.nearfield_dot(v))
        near_t = tsk.task(tsk_w, call_nearfield)
        far_t = tsk.task(tsk_w, call_farfield)
        far_out = await far_t
        near_out = await near_t
        return near_out + far_out

    def dot(self, v):
        async def wrapper(tsk_w):
            return await self.async_dot(tsk_w, v)
        return tsk.run(wrapper)

    async def farfield_dot(self, tsk_w, v):
        return (await self.farfield.async_dot(tsk_w, v))
