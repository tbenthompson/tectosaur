import os
import numpy as np
import pycuda.driver as drv
import scipy.sparse

from tectosaur.quadrature import richardson_quad, gauss4d_tri, gauss2d_tri
from tectosaur.adjacency import find_adjacents, vert_adj_prep,\
    edge_adj_prep, rotate_tri
import tectosaur.triangle_rules as triangle_rules
from tectosaur.util.gpu import load_gpu
from tectosaur.util.timer import Timer
from tectosaur.util.caching import cache


def pairs_func_name(singular, filtered_same_pt, k_name):
    singular_label = 'N'
    if singular:
        singular_label = 'S'
    filter_label = ''
    if filtered_same_pt:
        filter_label = 'F'
    return 'single_pairs' + singular_label + filter_label + k_name

def get_gpu_config():
    return {'block_size': 128}

def get_gpu_module():
    return load_gpu('tectosaur/integrals.cu', tmpl_args = get_gpu_config())

def get_pairs_integrator(singular, filtered_same_pt):
    return get_gpu_module().get_function(
        pairs_func_name(singular, filtered_same_pt, 'H')
    )

def pairs_quad(sm, pr, pts, obs_tris, src_tris, q, singular, filtered_same_pt):
    integrator = get_pairs_integrator(singular, filtered_same_pt)

    result = np.empty((obs_tris.shape[0], 3, 3, 3, 3)).astype(np.float32)
    if obs_tris.shape[0] == 0:
        return result

    block_main = (32, 1, 1)
    remaining = obs_tris.shape[0] % block_main[0]

    result_rem = np.empty((remaining, 3, 3, 3, 3)).astype(np.float32)

    grid_main = (obs_tris.shape[0] // block_main[0], 1, 1)
    grid_rem = (remaining, 1, 1)

    def call_integrator(block, grid, result_buf, tri_start, tri_end):
        if grid[0] == 0:
            return
        integrator(
            drv.Out(result_buf),
            np.int32(q[0].shape[0]),
            drv.In(q[0].astype(np.float32)),
            drv.In(q[1].astype(np.float32)),
            drv.In(pts.astype(np.float32)),
            drv.In(obs_tris[tri_start:tri_end].astype(np.int32)),
            drv.In(src_tris[tri_start:tri_end].astype(np.int32)),
            np.float32(sm),
            np.float32(pr),
            block = block, grid = grid
        )
    call_integrator(block_main, grid_main, result, 0, obs_tris.shape[0] - remaining)
    call_integrator(
        (1,1,1), grid_rem, result_rem,
        obs_tris.shape[0] - remaining, obs_tris.shape[0])
    result[obs_tris.shape[0] - remaining:,:,:,:,:] = result_rem
    return result

def farfield(sm, pr, pts, obs_tris, src_tris, n_q):
    q = gauss4d_tri(n_q)

    def call_integrator(block, grid, result_buf, tri_start, tri_end):
        integrator = get_gpu_module().get_function("farfield_trisH")
        if grid[0] == 0:
            return
        integrator(
            drv.Out(result_buf),
            np.int32(q[0].shape[0]),
            drv.In(q[0].astype(np.float32)),
            drv.In(q[1].astype(np.float32)),
            drv.In(pts.astype(np.float32)),
            np.int32(tri_end - tri_start),
            drv.In(obs_tris[tri_start:tri_end].astype(np.int32)),
            np.int32(src_tris.shape[0]),
            drv.In(src_tris.astype(np.int32)),
            np.float32(sm),
            np.float32(pr),
            block = block,
            grid = grid
        )

    max_block = 32
    max_grid = 20
    cur_result_buf = np.empty(
        (max_block * max_grid, 3, 3, src_tris.shape[0], 3, 3)
    ).astype(np.float32)

    full_result = np.empty(
        (obs_tris.shape[0], 3, 3, src_tris.shape[0], 3, 3)
    ).astype(np.float32)

    def next_integral(next_tri = 0):
        remaining = obs_tris.shape[0] - next_tri
        if remaining == 0:
            return
        elif remaining > max_block:
            block = (max_block, 1, 1)
        else:
            block = (remaining, 1, 1)
        grid = (min(remaining // block[0], max_grid), src_tris.shape[0])
        n_tris = grid[0] * block[0]
        past_end_tri = next_tri + n_tris
        call_integrator(block, grid, cur_result_buf, next_tri, past_end_tri)
        full_result[next_tri:past_end_tri] = cur_result_buf[:n_tris]
        next_integral(past_end_tri)

    next_integral()
    return full_result

def cached_in(name, creator):
    filename = os.path.join('cache_tectosaur', name + '.npy')
    if not os.path.exists(filename):
        dirname = os.path.dirname(filename)
        print(dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        np.save(filename, *creator())
    return np.load(filename)

@cache
def cached_coincident_quad(nq, eps):
    return richardson_quad(
        eps, lambda e: triangle_rules.coincident_quad(e, nq, nq, nq, nq)
    )

def coincident(nq, sm, pr, pts, tris):
    timer = Timer(2)
    q = cached_coincident_quad(nq, [0.1, 0.01])
    timer.report("Generate quadrature rule")
    out = pairs_quad(sm, pr, pts, tris, tris, q, True, False)
    timer.report("Perform quadrature")
    return out

@cache
def cached_edge_adj_quad(nq, eps):
    return richardson_quad(
        eps, lambda e: triangle_rules.edge_adj_quad(e, nq, nq, nq, nq, False)
    )

def edge_adj(nq, sm, pr, pts, obs_tris, src_tris):
    timer = Timer(2)
    q = cached_edge_adj_quad(nq, [0.1, 0.01])
    timer.report("Generate quadrature rule")
    out = pairs_quad(sm, pr, pts, obs_tris, src_tris, q, True, False)
    timer.report("Perform quadrature")
    return out

@cache
def cached_vert_adj_quad(nq):
    return triangle_rules.vertex_adj_quad(nq, nq, nq)

def vert_adj(nq, sm, pr, pts, obs_tris, src_tris):
    timer = Timer(2)
    q = cached_vert_adj_quad(nq)
    timer.report("Generate quadrature rule")
    out = pairs_quad(sm, pr, pts, obs_tris, src_tris, q, False, False)
    timer.report("Perform quadrature")
    return out

def set_co_entries(mat, co_mat, co_indices):
    mat[co_indices, :, :, co_indices, :, :] += co_mat
    return mat

def set_adj_entries(mat, adj_mat, tri_idxs, obs_clicks, src_clicks, corr_mat):
    obs_derot = rotate_tri(obs_clicks)
    src_derot = rotate_tri(src_clicks)

    placeholder = np.arange(adj_mat.shape[0])
    for b1 in range(3):
        for b2 in range(3):
            mat[tri_idxs[:,0], b1, :, tri_idxs[:,1], b2, :] += \
                adj_mat[placeholder, obs_derot[b1], :, src_derot[b2], :] -\
                corr_mat[placeholder, b1, :, b2, :]
    return mat

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

    basis = np.array([1 - qx[:, 0] - qx[:, 1], qx[:, 0], qx[:, 1]]).T

    unscaled_normals = np.cross(
        tri_pts[:,2,:] - tri_pts[:,0,:],
        tri_pts[:,2,:] - tri_pts[:,1,:]
    )
    jacobians = np.linalg.norm(unscaled_normals, axis = 1)
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

class SelfIntegralOperator:
    def __init__(self, nq_coincident, nq_edge_adjacent,
            nq_vert_adjacent, sm, pr, pts, tris):

        nq_far = 3
        far_quad = gauss4d_tri(nq_far)

        timer = Timer(tabs = 1)
        co_indices = np.arange(tris.shape[0])
        co_mat = coincident(nq_coincident, sm, pr, pts, tris)
        timer.report("Coincident")

        va, ea = find_adjacents(tris)
        timer.report("Find adjacency")

        ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_obs_tris, ea_src_tris =\
            edge_adj_prep(tris, ea)
        timer.report("Edge adjacency prep")
        ea_mat_rot = edge_adj(nq_edge_adjacent, sm, pr, pts, ea_obs_tris, ea_src_tris)
        timer.report("Edge adjacent")

        va_tri_indices, va_obs_clicks, va_src_clicks, va_obs_tris, va_src_tris =\
            vert_adj_prep(tris, va)
        timer.report("Vert adjacency prep")

        va_mat_rot = vert_adj(nq_vert_adjacent, sm, pr, pts, va_obs_tris, va_src_tris)
        timer.report("Vert adjacent")

        co_mat -= pairs_quad(
            sm, pr, pts, tris, tris, gauss4d_tri(3), False, True
        )
        ea_mat_corr = pairs_quad(
            sm, pr, pts,
            tris[ea_tri_indices[:,0]], tris[ea_tri_indices[:,1]],
            gauss4d_tri(3), False, True
        )
        va_mat_corr = pairs_quad(
            sm, pr, pts,
            tris[va_tri_indices[:,0]], tris[va_tri_indices[:,1]],
            gauss4d_tri(3), False, True
        )

        timer.report("farfield correction")

        def co_sparse_mat(co_indices, co_mat):
            tiled_co_indices = np.tile(co_indices[:,np.newaxis,np.newaxis], (1,3,3))
            indices = tiled_co_indices * 9 +\
                np.arange(3)[np.newaxis,:,np.newaxis] * 3 +\
                np.arange(3)[np.newaxis, np.newaxis, :]
            rows = np.tile(indices[:,:,:,np.newaxis,np.newaxis], (1,1,1,3,3)).flatten()
            cols = np.tile(indices[:,np.newaxis,np.newaxis], (1,3,3,1,1)).flatten()
            vals = co_mat.flatten()
            return vals, rows, cols

        def adj_sparse_mat(adj_mat, tri_idxs, obs_clicks, src_clicks, corr_mat):
            obs_derot = np.array(rotate_tri(obs_clicks))
            src_derot = np.array(rotate_tri(src_clicks))

            derot_mat = np.empty_like(adj_mat)
            placeholder = np.arange(adj_mat.shape[0])
            for b1 in range(3):
                for b2 in range(3):
                    derot_mat[placeholder,b1,:,b2,:] =\
                        adj_mat[placeholder,obs_derot[b1],:,src_derot[b2],:] -\
                        corr_mat[placeholder,b1,:,b2,:]

            rows = np.tile((
                np.tile(tri_idxs[:,0,np.newaxis,np.newaxis], (1,3,3)) * 9 +
                    np.arange(3)[np.newaxis,:,np.newaxis] * 3 +
                    np.arange(3)[np.newaxis, np.newaxis, :]
                )[:,:,:,np.newaxis,np.newaxis], (1,1,1,3,3)
            ).flatten()
            cols = np.tile((
                np.tile(tri_idxs[:,1,np.newaxis,np.newaxis], (1,3,3)) * 9 +
                    np.arange(3)[np.newaxis,:,np.newaxis] * 3 +
                    np.arange(3)[np.newaxis, np.newaxis, :]
                )[:,np.newaxis,np.newaxis], (1,3,3,1,1)
            ).flatten()

            vals = derot_mat.flatten()
            return vals, rows, cols

        co_vals,co_rows,co_cols = co_sparse_mat(co_indices, co_mat)
        ea_vals,ea_rows,ea_cols = adj_sparse_mat(
            ea_mat_rot, ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_mat_corr
        )
        va_vals,va_rows,va_cols = adj_sparse_mat(
            va_mat_rot, va_tri_indices, va_obs_clicks, va_src_clicks, va_mat_corr
        )
        rows = np.hstack((co_rows, ea_rows, va_rows))
        cols = np.hstack((co_cols, ea_cols, va_cols))
        vals = np.hstack((co_vals, ea_vals, va_vals))
        self.coincident = scipy.sparse.coo_matrix((vals, (rows, cols))).tocsr()

        timer.report("Insert coincident adjacent")

        far_quad2d = gauss2d_tri(nq_far)
        self.interp_galerkin_mat, self.quad_pts, self.quad_ns = \
            interp_galerkin_mat(pts[tris], far_quad2d)
        self.quad_pts = self.quad_pts.flatten().astype(np.float32)
        self.quad_ns = self.quad_ns.flatten().astype(np.float32)
        self.nq = int(self.quad_pts.shape[0] / 3)
        self.shape = self.coincident.shape
        self.sm = sm
        self.pr = pr
        self.gpu_module = get_gpu_module()

    def dot(self, v):
        t = Timer()
        out = self.coincident.dot(v)
        t.report('nearfield')

        interp_v = self.interp_galerkin_mat.dot(v).flatten().astype(np.float32)
        t.report('interp * v')
        nbody_result = np.empty((self.nq * 3), dtype = np.float32)
        t.report('empty nbody_result')
        block = (get_gpu_config()['block_size'], 1, 1)
        grid = (int(np.ceil(self.nq / block[0])), 1)
        runtime = self.gpu_module.get_function("farfield_ptsH")(
            drv.Out(nbody_result),
            drv.In(self.quad_pts), drv.In(self.quad_ns),
            drv.In(self.quad_pts), drv.In(self.quad_ns),
            drv.In(interp_v),
            np.float32(self.sm), np.float32(self.pr),
            np.int32(self.nq), np.int32(self.nq),
            block = block,
            grid = grid,
            time_kernel = True
        )
        t.report('call farfield interact')
        out += self.interp_galerkin_mat.T.dot(nbody_result)
        t.report('galerkin * nbody_result')
        return out
