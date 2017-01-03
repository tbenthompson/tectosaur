import os
import numpy as np
import scipy.sparse

from tectosaur.adjacency import find_adjacents, vert_adj_prep, \
    edge_adj_prep, rotate_tri
from tectosaur.find_nearfield import find_nearfield
from tectosaur.quadrature import gauss4d_tri, gauss2d_tri
import tectosaur.fmm as fmm
import tectosaur.geometry as geometry
#TODO: Split the cuda code into nearfield integrals and farfield.
from tectosaur.nearfield_op import coincident, pairs_quad, edge_adj, vert_adj,\
    get_ocl_gpu_module, get_gpu_config, float_type
from tectosaur.table_lookup import coincident_table, adjacent_table
import tectosaur.util.gpu as gpu

from tectosaur.util.timer import Timer


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
    rows = np.tile((
        np.tile(obs_idxs[:,np.newaxis,np.newaxis], (1,3,3)) * 9 +
            np.arange(3)[np.newaxis,:,np.newaxis] * 3 +
            np.arange(3)[np.newaxis, np.newaxis, :]
        )[:,:,:,np.newaxis,np.newaxis], (1,1,1,3,3)
    ).flatten()
    cols = np.tile((
        np.tile(src_idxs[:,np.newaxis,np.newaxis], (1,3,3)) * 9 +
            np.arange(3)[np.newaxis,:,np.newaxis] * 3 +
            np.arange(3)[np.newaxis, np.newaxis, :]
        )[:,np.newaxis,np.newaxis], (1,3,3,1,1)
    ).flatten()
    return integrals.flatten(), rows, cols

def co_sparse_mat(co_indices, co_mat, correction):
    return pairs_sparse_mat(co_indices, co_indices, co_mat - correction)

def derot_adj_mat(adj_mat, obs_clicks, src_clicks):
    # The triangles were rotated prior to performing the adjacent nearfield
    # quadrature. Therefore, the resulting matrix entries need to be
    # "derotated" in order to be consistent with the global basis function
    # numbering.

    obs_derot = np.array(rotate_tri(-obs_clicks))
    src_derot = np.array(rotate_tri(-src_clicks))

    derot_mat = np.empty_like(adj_mat)
    placeholder = np.arange(adj_mat.shape[0])
    for b1 in range(3):
        for b2 in range(3):
            derot_mat[placeholder,b1,:,b2,:] =\
                adj_mat[placeholder,obs_derot[b1],:,src_derot[b2],:]
    return derot_mat

def adj_sparse_mat(adj_mat, tri_idxs, obs_clicks, src_clicks, correction_mat):
    derot_mat = derot_adj_mat(adj_mat, obs_clicks, src_clicks)
    return pairs_sparse_mat(tri_idxs[:,0],tri_idxs[:,1], derot_mat - correction_mat)

def near_sparse_mat(near_mat, near_pairs, near_correction):
    return pairs_sparse_mat(near_pairs[:, 0], near_pairs[:, 1], near_mat - near_correction)

def build_nearfield(co_data, ea_data, va_data, near_data):
    co_vals,co_rows,co_cols = co_sparse_mat(*co_data)
    ea_vals,ea_rows,ea_cols = adj_sparse_mat(*ea_data)
    va_vals,va_rows,va_cols = adj_sparse_mat(*va_data)
    near_vals,near_rows,near_cols = near_sparse_mat(*near_data)
    rows = np.hstack((co_rows, ea_rows, va_rows, near_rows))
    cols = np.hstack((co_cols, ea_cols, va_cols, near_cols))
    vals = np.hstack((co_vals, ea_vals, va_vals, near_vals))
    return scipy.sparse.coo_matrix((vals, (rows, cols))).tocsr()

class NearfieldIntegralOp:
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, sm, pr, pts, tris,
            use_tables = False, remove_sing = False):
        near_gauss = gauss4d_tri(nq_near, nq_near)
        far_quad = gauss4d_tri(nq_far, nq_far)

        timer = Timer(tabs = 1, silent = True)
        co_indices = np.arange(tris.shape[0])
        if not use_tables:
            co_mat = coincident(nq_coincident, eps, kernel, sm, pr, pts, tris, remove_sing)
        else:
            co_mat = coincident_table(kernel, sm, pr, pts, tris, remove_sing)
        co_mat_correction = pairs_quad(
            kernel, sm, pr, pts, tris, tris, far_quad, False
        )
        timer.report("Coincident")

        va, ea = find_adjacents(tris)
        timer.report("Find adjacency")

        ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_obs_tris, ea_src_tris =\
            edge_adj_prep(tris, ea)
        timer.report("Edge adjacency prep")
        if not use_tables:
            ea_mat_rot = edge_adj(
                nq_edge_adjacent, eps, kernel, sm, pr, pts, ea_obs_tris, ea_src_tris, remove_sing
            )
        else:
            ea_mat_rot = adjacent_table(
                nq_vert_adjacent, kernel, sm, pr, pts, ea_obs_tris, ea_src_tris, remove_sing
            )
        ea_mat_correction = pairs_quad(
            kernel, sm, pr, pts,
            tris[ea_tri_indices[:,0]], tris[ea_tri_indices[:,1]],
            far_quad, False
        )
        timer.report("Edge adjacent")

        va_tri_indices, va_obs_clicks, va_src_clicks, va_obs_tris, va_src_tris =\
            vert_adj_prep(tris, va)
        timer.report("Vert adjacency prep")

        va_mat_rot = vert_adj(nq_vert_adjacent, kernel, sm, pr, pts, va_obs_tris, va_src_tris)
        va_mat_correction = pairs_quad(
            kernel, sm, pr, pts,
            tris[va_tri_indices[:,0]], tris[va_tri_indices[:,1]],
            far_quad, False
        )
        timer.report("Vert adjacent")

        nearfield_pairs = np.array(find_nearfield(pts, tris, va, ea, near_threshold))
        if nearfield_pairs.size == 0:
            nearfield_pairs = np.array([], dtype = np.int).reshape(0,2)
        timer.report("Find nearfield")

        nearfield_mat = pairs_quad(
            kernel, sm, pr, pts, tris[nearfield_pairs[:,0]], tris[nearfield_pairs[:, 1]],
            near_gauss, False
        )
        nearfield_correction = pairs_quad(
            kernel, sm, pr, pts, tris[nearfield_pairs[:,0]], tris[nearfield_pairs[:, 1]],
            far_quad, False
        )
        timer.report("Nearfield")

        self.mat = build_nearfield(
            (co_indices, co_mat, co_mat_correction),
            (ea_mat_rot, ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_mat_correction),
            (va_mat_rot, va_tri_indices, va_obs_clicks, va_src_clicks, va_mat_correction),
            (nearfield_mat, nearfield_pairs, nearfield_correction)
        )
        self.mat_no_correction = build_nearfield(
            (co_indices, co_mat, 0 * co_mat_correction),
            (ea_mat_rot, ea_tri_indices, ea_obs_clicks,
                ea_src_clicks, 0 * ea_mat_correction),
            (va_mat_rot, va_tri_indices, va_obs_clicks,
                va_src_clicks, 0 * va_mat_correction),
            (nearfield_mat, nearfield_pairs, 0 * nearfield_correction)
        )

    def dot(self, v):
        return self.mat.dot(v)

def farfield_pts_wrapper(K, n_obs, obs_pts, obs_ns, n_src, src_pts, src_ns, vec, sm, pr):
    gpu_farfield_fnc = getattr(get_ocl_gpu_module(), "farfield_pts" + K)

    gpu_result = gpu.empty_gpu(n_obs * 3, float_type)
    gpu_obs_pts = gpu.to_gpu(obs_pts, float_type)
    gpu_obs_ns = gpu.to_gpu(obs_ns, float_type)
    gpu_src_pts = gpu.to_gpu(src_pts, float_type)
    gpu_src_ns = gpu.to_gpu(src_ns, float_type)
    gpu_vec = gpu.to_gpu(vec, float_type)

    local_size = get_gpu_config()['block_size']
    n_blocks = int(np.ceil(n_obs / local_size))
    global_size = local_size * n_blocks
    gpu_farfield_fnc(
        gpu.ocl_gpu_queue, (global_size,), (local_size,),
        gpu_result.data,
        gpu_obs_pts.data, gpu_obs_ns.data,
        gpu_src_pts.data, gpu_src_ns.data,
        gpu_vec.data,
        float_type(sm), float_type(pr),
        np.int32(n_obs), np.int32(n_src),
    )
    return gpu_result.get()

class SparseIntegralOp:
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, sm, pr, pts, tris,
            use_tables = False, remove_sing = False):
        self.nearfield = NearfieldIntegralOp(
            eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, sm, pr, pts, tris,
            use_tables, remove_sing
        )

        far_quad2d = gauss2d_tri(nq_far)
        self.interp_galerkin_mat, quad_pts, quad_ns = \
            interp_galerkin_mat(pts[tris], far_quad2d)
        self.nq = quad_pts.shape[0]
        self.shape = self.nearfield.mat.shape
        self.sm = sm
        self.pr = pr
        self.kernel = kernel
        # self.gpu_farfield_fnc = getattr(get_ocl_gpu_module(), "farfield_pts" + kernel)
        self.gpu_quad_pts = gpu.to_gpu(quad_pts.flatten(), float_type)
        self.gpu_quad_ns = gpu.to_gpu(quad_ns.flatten(), float_type)

    def dot(self, v):
        out = self.nearfield.dot(v)
        return out + self.farfield_dot(v)

    def farfield_dot(self, v):
        interp_v = self.interp_galerkin_mat.dot(v).flatten()
        nbody_result = farfield_pts_wrapper(
            self.kernel, self.nq, self.gpu_quad_pts, self.gpu_quad_ns,
            self.nq, self.gpu_quad_pts, self.gpu_quad_ns, interp_v, self.sm, self.pr
        )
        out = self.interp_galerkin_mat.T.dot(nbody_result)
        return out

class FMMIntegralOp:
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, sm, pr, pts, tris):
        self.nearfield = NearfieldIntegralOp(
            eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, sm, pr, pts, tris
        )

        far_quad2d = gauss2d_tri(nq_far)
        self.interp_galerkin_mat, quad_pts, quad_ns = \
            interp_galerkin_mat(pts[tris], far_quad2d)
        self.shape = self.nearfield.mat.shape
        quad_ns = quad_ns.reshape(quad_pts.shape)

        order = 100
        mac = 3.0
        self.obs_kd = fmm.KDTree(quad_pts, quad_ns, order)
        self.src_kd = fmm.KDTree(quad_pts, quad_ns, order)
        self.fmm_mat = fmm.fmmmmmmm(
            self.obs_kd, self.src_kd,
            fmm.FMMConfig(1.1, mac, order, 'elastic' + kernel, [sm, pr])
        )

    def dot(self, v):
        out = self.nearfield.dot(v)
        out += self.farfield_dot(v)
        return out

    def farfield_dot(self, v):
        interp_v = self.interp_galerkin_mat.dot(v)
        fmm_out = fmm.eval(
            self.obs_kd, self.src_kd, self.fmm_mat, interp_v,
            interp_v.shape[0]
        )
        return self.interp_galerkin_mat.T.dot(fmm_out)

