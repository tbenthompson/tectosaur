import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

import tectosaur.triangle_rules as triangle_rules
from tectosaur.quadrature import richardson_quad
from tectosaur.util.gpu import load_gpu
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
        eps, lambda e: triangle_rules.coincident_quad(e, nq, nq, nq)
    )

def coincident(nq, eps, sm, pr, pts, tris):
    q = cached_coincident_quad(nq, eps)
    out = pairs_quad(sm, pr, pts, tris, tris, q, True, False)
    return out

@cache
def cached_edge_adj_quad(nq, eps):
    return richardson_quad(
        eps, lambda e: triangle_rules.edge_adj_quad(e, nq, nq, nq)
    )

def edge_adj(nq, eps, sm, pr, pts, obs_tris, src_tris):
    q = cached_edge_adj_quad(nq, eps)
    out = pairs_quad(sm, pr, pts, obs_tris, src_tris, q, True, False)
    return out

@cache
def cached_vert_adj_quad(nq):
    return triangle_rules.vertex_adj_quad(nq, nq, nq)

def vert_adj(nq, sm, pr, pts, obs_tris, src_tris):
    q = cached_vert_adj_quad(nq)
    out = pairs_quad(sm, pr, pts, obs_tris, src_tris, q, False, False)
    return out
