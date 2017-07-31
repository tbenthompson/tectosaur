import os
import numpy as np
import scipy.sparse


from tectosaur.farfield import farfield_pts_direct

from tectosaur.nearfield.nearfield_op import NearfieldIntegralOp
from tectosaur.nearfield.table_lookup import coincident_table, adjacent_table

from tectosaur.util.quadrature import gauss2d_tri
import tectosaur.util.geometry as geometry
import tectosaur.util.gpu as gpu
from tectosaur.util.timer import Timer

from tectosaur.util.logging import setup_logger
from tectosaur.util.build_cfg import float_type


logger = setup_logger(__name__)

from cppimport import cppimport
fast_assembly = cppimport("tectosaur.ops.fast_assembly")

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
    def __init__(self, kernel, params, obs_pts, obs_ns, src_pts, src_ns):
        self.params = params
        self.kernel = kernel
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

    def dot(self, v):
        return farfield_pts_direct(
            self.kernel, self.gpu_obs_pts, self.gpu_obs_ns,
            self.gpu_src_pts, self.gpu_src_ns, v, self.params
        )

class FMMFarfield:
    def __init__(self, kernel, params, obs_pts, obs_ns, src_pts, src_ns):
        import tectosaur.fmm.fmm_wrapper as fmm
        self.fmm_module = fmm
        order = 100
        mac = 3.0
        pts_per_cell = 200
        # TODO: different obs and src pts
        self.tree = fmm.three.Octree(obs_pts, pts_per_cell)
        self.fmm_mat = fmm.three.fmmmmmmm(
            self.tree, obs_ns[self.tree.orig_idxs],
            self.tree, src_ns[self.tree.orig_idxs],
            fmm.three.FMMConfig(1.1, mac, order, kernel, params)
        )
        fmm.report_interactions(self.fmm_mat)
        self.gpu_data = fmm.data_to_gpu(self.fmm_mat)
        self.orig_idxs = np.array(self.tree.orig_idxs)

    def dot(self, v):
        input_tree = v.reshape((-1,3))[self.orig_idxs,:].reshape(-1)
        fmm_out = self.fmm_module.eval_ocl(
            self.fmm_mat, input_tree, self.gpu_data
        ).reshape((-1, 3))
        to_orig = np.empty_like(fmm_out)
        to_orig[self.orig_idxs,:] = fmm_out
        return to_orig.reshape(-1)

class SparseIntegralOp:
    def __init__(self, nq_vert_adjacent, nq_far, nq_near,
            near_threshold, kernel, params, pts, tris, farfield_op_type = None):

        self.nearfield = NearfieldIntegralOp(
            nq_vert_adjacent, nq_far, nq_near,
            near_threshold, kernel, params, pts, tris
        )

        far_quad2d = gauss2d_tri(nq_far)
        self.interp_galerkin_mat, quad_pts, quad_ns = \
            interp_galerkin_mat(pts[tris], far_quad2d)
        self.shape = self.nearfield.shape

        if farfield_op_type is None:
            farfield_op_type = DirectFarfield

        self.farfield_op = farfield_op_type(
            kernel, params, quad_pts, quad_ns, quad_pts, quad_ns
        )

    def nearfield_dot(self, v):
        return self.nearfield.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.nearfield.mat_no_correction.dot(v)

    def dot(self, v):
        near_out = self.nearfield.dot(v)
        far_out = self.farfield_dot(v)
        return near_out + far_out

    def farfield_dot(self, v):
        interp_v = self.interp_galerkin_mat.dot(v).flatten()
        nbody_result = self.farfield_op.dot(interp_v)
        out = self.interp_galerkin_mat.T.dot(nbody_result)
        return out
