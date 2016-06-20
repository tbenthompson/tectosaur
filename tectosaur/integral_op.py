import numpy as np
import pycuda.driver as drv

from tectosaur.quadrature import richardson_quad, gauss4d_tri
from tectosaur.adjacency import find_adjacents, vert_adj_prep,\
    edge_adj_prep, rotate_tri
from tectosaur.gpu import load_gpu
import tectosaur.triangle_rules as triangle_rules
from tectosaur.timer import Timer

gpu_module = load_gpu('tectosaur/integrals.cu')
def get_pairs_integrator(singular):
    if singular:
        integrator = gpu_module.get_function('single_pairsSH')
    else:
        integrator = gpu_module.get_function('single_pairsNH')
    return integrator;


def pairs_quad(sm, pr, pts, obs_tris, src_tris, q, singular):
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
    integrator = gpu_module.get_function("farfield_trisH")
    q = gauss4d_tri(n_q)

    result = np.empty(
        (obs_tris.shape[0], src_tris.shape[0], 3, 3, 3, 3)
    ).astype(np.float32)

    block_main = (32, 1, 1)
    remaining = obs_tris.shape[0] % block_main[0]

    result_rem = np.empty(
        (remaining, src_tris.shape[0], 3, 3, 3, 3)
    ).astype(np.float32)

    grid_main = (obs_tris.shape[0] // block_main[0], src_tris.shape[0])
    grid_rem = (remaining, src_tris.shape[0])

    def call_integrator(block, grid, result_buf, tri_start, tri_end):
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
    call_integrator(block_main, grid_main, result, 0, obs_tris.shape[0] - remaining)
    call_integrator(
        (1,1,1), grid_rem, result_rem,
        obs_tris.shape[0] - remaining, obs_tris.shape[0]
    )
    result[-remaining:,:,:,:,:,:] = result_rem
    return result

def coincident(nq, sm, pr, pts, tris):
    timer = Timer(2)
    q = richardson_quad(
        [0.1, 0.01],
        lambda e: triangle_rules.coincident_quad(e, nq, nq, nq, nq)
    )
    from IPython import embed; embed(); import ipdb; ipdb.set_trace()
    timer.report("Generate quadrature rule")
    out = pairs_quad(sm, pr, pts, tris, tris, q, True)
    timer.report("Perform quadrature")
    return out

def edge_adj(nq, sm, pr, pts, obs_tris, src_tris):
    timer = Timer(2)
    q = richardson_quad(
        [0.1, 0.01],
        lambda e: triangle_rules.edge_adj_quad(e, nq, nq, nq, nq, False)
    )
    timer.report("Generate quadrature rule")
    out = pairs_quad(sm, pr, pts, obs_tris, src_tris, q, True)
    timer.report("Perform quadrature")
    return out

def vert_adj(nq, sm, pr, pts, obs_tris, src_tris):
    timer = Timer(2)
    q = triangle_rules.vertex_adj_quad(nq, nq, nq)
    timer.report("Generate quadrature rule")
    out = pairs_quad(sm, pr, pts, obs_tris, src_tris, q, False)
    timer.report("Perform quadrature")
    return out

def set_adj_entries(mat, adj_mat, tri_idxs, obs_clicks, src_clicks):
    for i in range(adj_mat.shape[0]):
        obs_derot = rotate_tri(obs_clicks[i])
        src_derot = rotate_tri(src_clicks[i])
        for b1 in range(3):
            for b2 in range(3):
                mat[tri_idxs[i,0], tri_idxs[i,1], b1, b2, :, :] =\
                    adj_mat[i, obs_derot[b1], src_derot[b2], :, :]
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

    far_mat = farfield(sm, pr, pts, tris, tris, 3)
    timer.report("Farfield")
    far_mat[co_indices, co_indices] = co_mat
    far_mat = set_adj_entries(
        far_mat, ea_mat_rot, ea_tri_indices, ea_obs_clicks, ea_src_clicks
    )
    far_mat = set_adj_entries(
        far_mat, va_mat_rot, va_tri_indices, va_obs_clicks, va_src_clicks
    )
    timer.report("Insert coincident adjacent")

    return far_mat
