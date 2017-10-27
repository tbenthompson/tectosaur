import os
import attr
import numpy as np
import scipy.sparse


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

    async def dot(self, v):
        return farfield_pts_direct(
            self.kernel, self.gpu_obs_pts, self.gpu_obs_ns,
            self.gpu_src_pts, self.gpu_src_ns, v, self.params, self.float_type
        )

class FMMFarfield:
    def __init__(self, kernel, params, obs_pts, obs_ns, src_pts, src_ns,
            float_type, order, mac, pts_per_cell):
        cfg = fmm.make_config(kernel, params, 1.1, mac, order, float_type)

        # TODO: different obs and src pts
        self.tree = fmm.make_tree(obs_pts.copy(), cfg, pts_per_cell)
        self.orig_idxs = np.array(self.tree.orig_idxs)
        self.fmm_obj = fmm.FMM(
            self.tree, obs_ns[self.orig_idxs].copy(),
            self.tree, src_ns[self.orig_idxs].copy(), cfg
        )
        fmm.report_interactions(self.fmm_obj)
        self.evaluator = fmm.FMMEvaluator(self.fmm_obj)

    async def dot(self, v):
        t = Timer()
        input_tree = v.reshape((-1,3))[self.orig_idxs,:].reshape(-1)
        t.report('to tree space')

        fmm_out = await self.evaluator.eval(input_tree.copy())
        fmm_out = fmm_out.reshape((-1, 3))
        t.report('fmm eval')

        to_orig = np.empty_like(fmm_out)
        to_orig[self.orig_idxs,:] = fmm_out
        to_orig = to_orig.flatten()
        t.report('to output space')
        return to_orig

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

import asyncio
class SparseIntegralOp:
    def __init__(self, nq_vert_adjacent, nq_far, nq_near, near_threshold,
            kernel, params, pts, tris, float_type, farfield_op_type = None):

        self.nearfield = NearfieldIntegralOp(
            nq_vert_adjacent, nq_far, nq_near,
            near_threshold, kernel, params, pts, tris, float_type
        )

        far_quad2d = gauss2d_tri(nq_far)
        self.interp_galerkin_mat, quad_pts, quad_ns = \
            interp_galerkin_mat(pts[tris], far_quad2d)
        self.interp_galerkin_mat = self.interp_galerkin_mat.tocsr()
        self.shape = self.nearfield.shape

        if farfield_op_type is None:
            farfield_op_type = DirectFarfield

        self.farfield_op = farfield_op_type(
            kernel, params, quad_pts, quad_ns, quad_pts, quad_ns, float_type
        )
        self.gpu_nearfield = None

    async def nearfield_dot(self, v):
        print("STARTNEAR")
        return self.nearfield.dot(v)

    async def nearfield_no_correction_dot(self, v):
        return self.nearfield.nearfield_no_correction_dot(v)

    async def async_dot(self, v):
        # far_out = await self.farfield_dot(v)
        # near_out = await self.nearfield_dot(v)
        far_out, near_out = await asyncio.gather(
            asyncio.ensure_future(self.farfield_dot(v)),
            asyncio.ensure_future(self.nearfield_dot(v))
        )
        return near_out + far_out

    def dot(self, v):
        return asyncio.get_event_loop().run_until_complete(self.async_dot(v))

    async def farfield_dot(self, v):
        print("STARTFAR")
        interp_v = self.interp_galerkin_mat.dot(v).flatten()
        nbody_result = await self.farfield_op.dot(interp_v)
        out = self.interp_galerkin_mat.T.dot(nbody_result)
        print("ENDFAR")
        return out
