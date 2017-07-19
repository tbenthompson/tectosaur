import os
import numpy as np
import scipy.sparse

from tectosaur.util.quadrature import gauss2d_tri

from tectosaur.farfield import farfield_pts_direct

from tectosaur.nearfield.nearfield_op import NearfieldIntegralOp
from tectosaur.nearfield.table_lookup import coincident_table, adjacent_table

import tectosaur.util.geometry as geometry
import tectosaur.util.gpu as gpu

from tectosaur import float_type

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
    quad_ns = np.tile(scaled_normals[:,np.newaxis,:], (1, nq, 1))

    return scipy.sparse.coo_matrix((vals, (rows, cols))), quad_pts, quad_ns

class SparseIntegralOp:
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, params, pts, tris,
            use_tables = False, remove_sing = False):
        self.nearfield = NearfieldIntegralOp(
            eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, params, pts, tris,
            use_tables, remove_sing
        )

        far_quad2d = gauss2d_tri(nq_far)
        self.interp_galerkin_mat, quad_pts, quad_ns = \
            interp_galerkin_mat(pts[tris], far_quad2d)
        self.nq = quad_pts.shape[0]
        self.shape = self.nearfield.shape
        self.params = params
        self.kernel = kernel
        self.gpu_quad_pts = gpu.to_gpu(quad_pts, float_type)
        self.gpu_quad_ns = gpu.to_gpu(quad_ns, float_type)

    def nearfield_dot(self, v):
        return self.nearfield.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.nearfield.mat_no_correction.dot(v)

    def dot(self, v):
        return self.nearfield.dot(v) + self.farfield_dot(v)

    def farfield_dot(self, v):
        interp_v = self.interp_galerkin_mat.dot(v).flatten()
        nbody_result = farfield_pts_direct(
            self.kernel, self.gpu_quad_pts, self.gpu_quad_ns,
            self.gpu_quad_pts, self.gpu_quad_ns, interp_v, self.params
        )
        out = self.interp_galerkin_mat.T.dot(nbody_result)
        return out
