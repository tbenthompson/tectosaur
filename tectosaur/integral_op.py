import os
import numpy as np
import pycuda.driver as drv

from tectosaur.quadrature import richardson_quad, gauss4d_tri
from tectosaur.adjacency import find_adjacents, vert_adj_prep,\
    edge_adj_prep, rotate_tri
import tectosaur.triangle_rules as triangle_rules
from tectosaur.util.gpu import load_gpu
from tectosaur.util.timer import Timer
from tectosaur.util.caching import cache


gpu_module = load_gpu('tectosaur/integrals.cu', tmpl_args = {'block_size': 32})
def get_pairs_integrator(singular):
    if singular:
        integrator = gpu_module.get_function('single_pairsSH')
    else:
        integrator = gpu_module.get_function('single_pairsNH')
    return integrator;


def pairs_quad(sm, pr, pts, obs_tris, src_tris, q, singular):
    print(obs_tris.shape[0])
    print(q[0].shape)
    print(obs_tris.shape[0])
    print(q[0].shape)
    print(obs_tris.shape[0])
    print(q[0].shape)
    print(obs_tris.shape[0])
    print(q[0].shape)
    integrator = get_pairs_integrator(singular)

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
        integrator = gpu_module.get_function("farfield_trisH")
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
    q = cached_coincident_quad(7, [0.1, 0.01])
    timer.report("Generate quadrature rule")
    out = pairs_quad(sm, pr, pts, tris, tris, q, True)
    timer.report("Perform quadrature")
    import sys
    sys.exit()
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
    out = pairs_quad(sm, pr, pts, obs_tris, src_tris, q, True)
    timer.report("Perform quadrature")
    return out

@cache
def cached_vert_adj_quad(nq):
    return triangle_rules.vertex_adj_quad(nq, nq, nq)

def vert_adj(nq, sm, pr, pts, obs_tris, src_tris):
    timer = Timer(2)
    q = cached_vert_adj_quad(nq)
    timer.report("Generate quadrature rule")
    out = pairs_quad(sm, pr, pts, obs_tris, src_tris, q, False)
    timer.report("Perform quadrature")
    return out

def set_co_entries(mat, co_mat, co_indices):
    mat[co_indices, :, :, co_indices, :, :] =\
        np.swapaxes(np.swapaxes(np.swapaxes(co_mat, 1, 3), 2, 3), 3, 4)
    return mat

def set_adj_entries(mat, adj_mat, tri_idxs, obs_clicks, src_clicks):
    obs_derot = rotate_tri(obs_clicks)
    src_derot = rotate_tri(src_clicks)

    placeholder = np.arange(adj_mat.shape[0])
    for b1 in range(3):
        for b2 in range(3):
            mat[tri_idxs[:,0], :, b1,tri_idxs[:,1], :, b2] = \
                adj_mat[placeholder, obs_derot[b1], src_derot[b2], :, :]
    return mat

def self_integral_operator(nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
                           sm, pr, pts, tris):
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

    out = farfield(sm, pr, pts, tris, tris, 3)
    timer.report("Farfield")
    out = set_co_entries(out, co_mat, co_indices)
    out = set_adj_entries(
        out, ea_mat_rot, ea_tri_indices, ea_obs_clicks, ea_src_clicks
    )
    out = set_adj_entries(
        out, va_mat_rot, va_tri_indices, va_obs_clicks, va_src_clicks
    )
    timer.report("Insert coincident adjacent")

    out.shape = (
        out.shape[0] * out.shape[1] * out.shape[2],
        out.shape[3] * out.shape[4] * out.shape[5]
    )
    return out
