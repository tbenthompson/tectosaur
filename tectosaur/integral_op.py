import numpy as np
import pycuda.driver as drv

from tectosaur.quadrature import richardson_quad, gauss4d_tri
from tectosaur.adjacency import find_adjacents, vert_adj_prep, edge_adj_prep
from tectosaur.gpu import load_gpu
import tectosaur.triangle_rules as triangle_rules

gpu_module = load_gpu('tectosaur/integrals.cu')
def pairs_quad(sm, pr, pts, obs_tris, src_tris, q, singular):
    if singular:
        integrator = gpu_module.get_function('single_pairsSH')
    else:
        integrator = gpu_module.get_function('single_pairsNH')

    result = np.empty((obs_tris.shape[0], 3, 3, 3, 3)).astype(np.float32)
    if obs_tris.shape[0] == 0:
        return result

    qx = q[0].astype(np.float32)
    qw = q[1].astype(np.float32)
    #TODO: Separate into stuff that can blocked into 32,1,1 and the
    #remainder that is done separately.
    block = (1, 1, 1)
    grid = (obs_tris.shape[0] // block[0], 1, 1)

    integrator(
        drv.Out(result),
        np.int32(qx.shape[0]), drv.In(qx), drv.In(qw),
        drv.In(pts.astype(np.float32)),
        drv.In(obs_tris.astype(np.int32)),
        drv.In(src_tris.astype(np.int32)),
        np.float32(sm), np.float32(pr),
        block = block, grid = grid
    )

    return result

def farfield(sm, pr, pts, tris, n_q):
    integrator = gpu_module.get_function("farfield_trisH")

    q = gauss4d_tri(n_q)
    qx = q[0].astype(np.float32)
    qw = q[1].astype(np.float32)

    result = np.empty((tris.shape[0], tris.shape[0], 3, 3, 3, 3)).astype(np.float32)

    #TODO: Separate into stuff that can blocked into 32,1,1 and the
    #remainder that is done separately.
    n_tris = tris.shape[0]
    block = (1, 1, 1)
    grid = (n_tris // block[0], n_tris // block[1])

    integrator(
        drv.Out(result),
        np.int32(qx.shape[0]),
        drv.In(qx),
        drv.In(qw),
        drv.In(pts),
        np.int32(tris.shape[0]),
        drv.In(tris),
        np.int32(tris.shape[0]),
        drv.In(tris),
        np.float32(sm),
        np.float32(pr),
        block = block,
        grid = grid,
    )

    return result

def coincident(sm, pr, pts, tris):
    q = richardson_quad(
        [0.1, 0.01],
        lambda e: triangle_rules.coincident_quad(e, 8, 8, 5, 10)
    )
    return pairs_quad(sm, pr, pts, tris, tris, q, True)

def edge_adj(sm, pr, pts, obs_tris, src_tris):
    nq = 8
    q = richardson_quad(
        [0.1, 0.01],
        lambda e: triangle_rules.edge_adj_quad(e, 8, 8, 8, 8, False)
    )
    return pairs_quad(sm, pr, pts, obs_tris, src_tris, q, True)

def vert_adj(sm, pr, pts, obs_tris, src_tris):
    nq = 3
    q = triangle_rules.vertex_adj_quad(nq, nq, nq)
    return pairs_quad(sm, pr, pts, obs_tris, src_tris, q, False)

def self_integral_operator(sm, pr, pts, tris):
    # co_mat = coincident(sm, pr, pts, tris)

    # va, ea = find_adjacents(tris)

    # ea_tri_indices, ea_obs_tris, ea_src_tris = edge_adj_prep(tris, ea)
    # ea_mat = edge_adj(sm, pr, pts, ea_obs_tris, ea_src_tris)

    # va_tri_indices, va_obs_tris, va_src_tris = vert_adj_prep(tris, va)
    # va_mat = vert_adj(sm, pr, pts, va_obs_tris, va_src_tris)

    far_mat = farfield(sm, pr, pts, tris, 3)
    import ipdb; ipdb.set_trace()

    # Outline
    # coincident
    # edge adjacent
    # -- determine from topology(CHECK)
    # -- rotate triangles so that src vtx 0 = obs vtx 1 and src vtx 1 = obs vtx 0(CHECK)
    # vertex adjacent
    # -- determine from topology (CHECK)
    # -- rotate triangles so that src vtx 0 = obs vtx 0(CHECK)
    # nearfield
    # -- sphere tree distances
    # -- use 3 point gauss?
    # subtract farfield correction
    # -- same quadrature as for farfield
    # farfield
    # -- use 2 point gauss? or 3 point tri rule?
