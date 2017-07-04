import os
import numpy as np
import scipy.sparse

from tectosaur.mesh.adjacency import find_adjacents, vert_adj_prep, \
    edge_adj_prep, rotate_tri
from tectosaur.mesh.find_nearfield import find_nearfield
from tectosaur.util.quadrature import gauss4d_tri, gauss2d_tri
import tectosaur.util.geometry as geometry

#TODO: Split the cuda code into nearfield integrals and farfield.
from tectosaur.nearfield.vert_adj import coincident, pairs_quad, edge_adj, vert_adj,\
    get_gpu_module, get_gpu_config, float_type
from tectosaur.nearfield.table_lookup import coincident_table, adjacent_table
import tectosaur.util.gpu as gpu

from tectosaur.util.timer import Timer

import cppimport
fast_assembly = cppimport.imp("tectosaur.ops.fast_assembly").ops.fast_assembly

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


def pairs_sparse_mat(obs_idxs, src_idxs, integrals):
    return integrals.reshape((-1, 9, 9)), obs_idxs, src_idxs

def co_sparse_mat(co_indices, co_mat, correction):
    return pairs_sparse_mat(co_indices, co_indices, co_mat - correction)

def adj_sparse_mat(adj_mat, tri_idxs, obs_clicks, src_clicks, correction_mat):
    adj_mat = adj_mat.astype(np.float32)
    fast_assembly.derotate_adj_mat(adj_mat, obs_clicks, src_clicks)
    return pairs_sparse_mat(tri_idxs[:,0],tri_idxs[:,1], adj_mat - correction_mat)

def near_sparse_mat(near_mat, near_pairs, near_correction):
    return pairs_sparse_mat(near_pairs[:, 0], near_pairs[:, 1], near_mat - near_correction)

def build_nearfield(co_data, ea_data, va_data, near_data, shape):
    t = Timer(tabs = 2)
    co_vals,co_rows,co_cols = co_sparse_mat(*co_data)
    ea_vals,ea_rows,ea_cols = adj_sparse_mat(*ea_data)
    va_vals,va_rows,va_cols = adj_sparse_mat(*va_data)
    near_vals,near_rows,near_cols = near_sparse_mat(*near_data)
    t.report("build pairs")
    rows = np.concatenate((co_rows, ea_rows, va_rows, near_rows))
    cols = np.concatenate((co_cols, ea_cols, va_cols, near_cols))
    vals = np.concatenate((co_vals, ea_vals, va_vals, near_vals))
    t.report("stack pairs")

    data, indices, indptr = fast_assembly.make_bsr_matrix(
        shape[0], shape[1], 9, 9, vals, rows, cols
    )
    t.report("to bsr")
    mat = scipy.sparse.bsr_matrix((data, indices, indptr))
    t.report('make bsr')

    return mat

class NearfieldIntegralOp:
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, params, pts, tris,
            use_tables = False, remove_sing = False):
        n = tris.shape[0] * 9
        self.shape = (n, n)

        timer = Timer(tabs = 1)
        near_gauss = gauss4d_tri(nq_near, nq_near)
        far_quad = gauss4d_tri(nq_far, nq_far)
        timer.report("Build gauss rules")

        co_indices = np.arange(tris.shape[0])
        if not use_tables:
            co_mat = coincident(nq_coincident, eps, kernel, params, pts, tris, remove_sing)
        else:
            co_mat = coincident_table(kernel, params, pts, tris)
        timer.report("Coincident")
        co_mat_correction = pairs_quad(
            kernel, params, pts, tris, tris, far_quad, False, True
        )
        timer.report("Coincident correction")

        va, ea = find_adjacents(tris)
        timer.report("Find adjacency")

        ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_obs_tris, ea_src_tris =\
            edge_adj_prep(tris, ea)
        timer.report("Edge adjacency prep")
        if not use_tables:
            ea_mat_rot = edge_adj(
                nq_edge_adjacent, eps, kernel, params, pts,
                ea_obs_tris, ea_src_tris, remove_sing
            )
        else:
            ea_mat_rot = adjacent_table(
                nq_vert_adjacent, kernel, params, pts, ea_obs_tris, ea_src_tris
            )
        timer.report("Edge adjacent")
        ea_mat_correction = pairs_quad(
            kernel, params, pts,
            tris[ea_tri_indices[:,0]], tris[ea_tri_indices[:,1]],
            far_quad, False, False
        )
        timer.report("Edge adjacent correction")

        va_tri_indices, va_obs_clicks, va_src_clicks, va_obs_tris, va_src_tris =\
            vert_adj_prep(tris, va)
        timer.report("Vert adjacency prep")

        va_mat_rot = vert_adj(
            nq_vert_adjacent, kernel, params, pts, va_obs_tris, va_src_tris
        )
        timer.report("Vert adjacent")
        va_mat_correction = pairs_quad(
            kernel, params, pts,
            tris[va_tri_indices[:,0]], tris[va_tri_indices[:,1]],
            far_quad, False, False
        )
        timer.report("Vert adjacent correction")

        nearfield_pairs = np.array(find_nearfield(pts, tris, va, ea, near_threshold))
        if nearfield_pairs.size == 0:
            nearfield_pairs = np.array([], dtype = np.int).reshape(0,2)
        timer.report("Find nearfield")

        nearfield_mat = pairs_quad(
            kernel, params, pts, tris[nearfield_pairs[:,0]], tris[nearfield_pairs[:, 1]],
            near_gauss, False, False
        )
        timer.report("Nearfield")
        nearfield_correction = pairs_quad(
            kernel, params, pts, tris[nearfield_pairs[:,0]], tris[nearfield_pairs[:, 1]],
            far_quad, False, False
        )
        timer.report("Nearfield correction")

        self.mat = build_nearfield(
            (co_indices, co_mat, co_mat_correction),
            (ea_mat_rot, ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_mat_correction),
            (va_mat_rot, va_tri_indices, va_obs_clicks, va_src_clicks, va_mat_correction),
            (nearfield_mat, nearfield_pairs, nearfield_correction),
            self.shape
        )
        timer.report("Assemble matrix")
        self.mat_no_correction = build_nearfield(
            (co_indices, co_mat, 0 * co_mat_correction),
            (ea_mat_rot, ea_tri_indices, ea_obs_clicks,
                ea_src_clicks, 0 * ea_mat_correction),
            (va_mat_rot, va_tri_indices, va_obs_clicks,
                va_src_clicks, 0 * va_mat_correction),
            (nearfield_mat, nearfield_pairs, 0 * nearfield_correction),
            self.shape
        )
        timer.report("Assemble uncorrected matrix")

    def dot(self, v):
        return self.mat.dot(v)

def farfield_pts_wrapper(K, obs_pts, obs_ns, src_pts, src_ns, vec, params):
    gpu_farfield_fnc = getattr(get_gpu_module(), "farfield_pts" + K)

    n_obs = obs_pts.shape[0]
    if len(obs_pts.shape) == 1:
        n_obs //= 3
    n_src = src_pts.shape[0]
    if len(src_pts.shape) == 1:
        n_src //= 3

    gpu_result = gpu.empty_gpu(n_obs * 3, float_type)
    gpu_obs_pts = gpu.to_gpu(obs_pts, float_type)
    gpu_obs_ns = gpu.to_gpu(obs_ns, float_type)
    gpu_src_pts = gpu.to_gpu(src_pts, float_type)
    gpu_src_ns = gpu.to_gpu(src_ns, float_type)
    gpu_vec = gpu.to_gpu(vec, float_type)
    gpu_params = gpu.to_gpu(np.array(params), float_type)

    local_size = get_gpu_config()['block_size']
    n_blocks = int(np.ceil(n_obs / local_size))
    global_size = local_size * n_blocks
    gpu_farfield_fnc(
        gpu.gpu_queue, (global_size,), (local_size,),
        gpu_result.data,
        gpu_obs_pts.data, gpu_obs_ns.data,
        gpu_src_pts.data, gpu_src_ns.data,
        gpu_vec.data,
        gpu_params.data,
        np.int32(n_obs), np.int32(n_src),
    )
    return gpu_result.get()

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
        self.gpu_quad_pts = gpu.to_gpu(quad_pts.flatten(), float_type)
        self.gpu_quad_ns = gpu.to_gpu(quad_ns.flatten(), float_type)

    def nearfield_dot(self, v):
        return self.nearfield.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.nearfield.mat_no_correction.dot(v)

    def dot(self, v):
        return self.nearfield.dot(v) + self.farfield_dot(v)

    def farfield_dot(self, v):
        interp_v = self.interp_galerkin_mat.dot(v).flatten()
        nbody_result = farfield_pts_wrapper(
            self.kernel, self.gpu_quad_pts, self.gpu_quad_ns,
            self.gpu_quad_pts, self.gpu_quad_ns, interp_v, self.params
        )
        out = self.interp_galerkin_mat.T.dot(nbody_result)
        return out
