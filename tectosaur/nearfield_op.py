import os
import numpy as np

import tectosaur.triangle_rules as triangle_rules
from tectosaur.quadrature import richardson_quad
import tectosaur.util.gpu as gpu
from tectosaur.util.caching import cache
from tectosaur.integral_utils import pairs_func_name

def get_gpu_config():
    return {'block_size': 128, 'float_type': gpu.np_to_c_type(float_type)}

def get_gpu_module():
    return gpu.load_gpu('integrals.cl', tmpl_args = get_gpu_config())

def get_pairs_integrator(kernel, singular):
    return getattr(get_gpu_module(), pairs_func_name(singular, kernel))

#TODO: One universal float type for all of tectosaur? tectosaur.config?
#TODO: The general structure of this caller is similar to the other in tectosaur_tables and sparse_integral_op and dense_integral_op.
float_type = np.float32
def pairs_quad(kernel, sm, pr, pts, obs_tris, src_tris, q, singular):
    integrator = get_pairs_integrator(kernel, singular)

    gpu_qx, gpu_qw = gpu.quad_to_gpu(q, float_type)

    n = obs_tris.shape[0]
    out = np.empty((n, 3, 3, 3, 3), dtype = float_type)
    if n == 0:
        return out

    gpu_pts = gpu.to_gpu(pts, float_type)

    def call_integrator(start_idx, end_idx):
        n_items = end_idx - start_idx
        gpu_result = gpu.empty_gpu((n_items, 3, 3, 3, 3), float_type)
        gpu_obs_tris = gpu.to_gpu(obs_tris[start_idx:end_idx], np.int32)
        gpu_src_tris = gpu.to_gpu(src_tris[start_idx:end_idx], np.int32)
        integrator(
            gpu.gpu_queue, (n_items,), None,
            gpu_result.data,
            np.int32(q[0].shape[0]), gpu_qx.data, gpu_qw.data,
            gpu_pts.data, gpu_obs_tris.data, gpu_src_tris.data,
            float_type(sm), float_type(pr),
        )
        out[start_idx:end_idx] = gpu_result.get()

    call_size = 2 ** 7
    for I in gpu.intervals(n, call_size):
        call_integrator(*I)
    return out

@cache
def cached_coincident_quad(nq, eps, remove_sing):
    if type(nq) is int:
        nq = (nq, nq, nq, nq)
    return richardson_quad(
        eps, remove_sing,
        lambda e: triangle_rules.coincident_quad(e, nq[0], nq[1], nq[2], nq[3])
    )

def coincident(nq, eps, kernel, sm, pr, pts, tris, remove_sing):
    q = cached_coincident_quad(nq, eps, remove_sing)
    out = pairs_quad(kernel, sm, pr, pts, tris, tris, q, True)
    return out

@cache
def cached_edge_adj_quad(nq, eps, remove_sing):
    if type(nq) is int:
        nq = (nq, nq, nq, nq)
    return richardson_quad(
        eps, remove_sing,
        lambda e: triangle_rules.edge_adj_quad(e, nq[0], nq[1], nq[2], nq[3], False)
    )

def edge_adj(nq, eps, kernel, sm, pr, pts, obs_tris, src_tris, remove_sing):
    q = cached_edge_adj_quad(nq, eps, remove_sing)
    out = pairs_quad(kernel, sm, pr, pts, obs_tris, src_tris, q, True)
    return out

@cache
def cached_vert_adj_quad(nq):
    if type(nq) is int:
        nq = (nq, nq, nq)
    return triangle_rules.vertex_adj_quad(nq[0], nq[1], nq[2])

def vert_adj(nq, kernel, sm, pr, pts, obs_tris, src_tris):
    q = cached_vert_adj_quad(nq)
    out = pairs_quad(kernel, sm, pr, pts, obs_tris, src_tris, q, False)
    return out
