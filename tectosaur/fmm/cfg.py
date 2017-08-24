import numpy as np
import attr

import tectosaur.util.gpu as gpu
from tectosaur.kernels import kernels

from tectosaur.fmm.surrounding_surf import surrounding_surf

from cppimport import cppimport
traversal_ext = cppimport("tectosaur.fmm.traversal_wrapper")

def get_dim_module(dim):
    return traversal_ext.two if dim == 2 else traversal_ext.three

def get_traversal_module(K):
    if type(K.scale_type) is int:
        return get_dim_module(K.spatial_dim).kdtree
    else:
        return get_dim_module(K.spatial_dim).octree

def get_gpu_module(surf, K, float_type, n_workers_per_block, n_c2e_block_rows):
    args = dict(
        n_workers_per_block = n_workers_per_block,
        n_c2e_block_rows = n_c2e_block_rows,
        gpu_float_type = gpu.np_to_c_type(float_type),
        surf = surf,
        K = K
    )
    gpu_module = gpu.load_gpu(
        'fmm/gpu_kernels.cl',
        tmpl_args = args
    )
    return gpu_module

@attr.s
class FMMConfig:
    K = attr.ib()
    params = attr.ib()
    surf = attr.ib()
    outer_r = attr.ib()
    inner_r = attr.ib()
    order = attr.ib()
    float_type = attr.ib()
    gpu_module = attr.ib()
    traversal_module = attr.ib()
    n_workers_per_block = attr.ib()
    n_c2e_block_rows = attr.ib()

def make_config(K_name, params, inner_r, outer_r, order,
        float_type, n_workers_per_block = 64, n_c2e_block_rows = 16):
    K = kernels[K_name]
    surf = surrounding_surf(order, K.spatial_dim)
    return FMMConfig(
        K = K,
        params = np.array(params),
        surf = surf,
        outer_r = outer_r,
        inner_r = inner_r,
        order = order,
        float_type = float_type,
        gpu_module = get_gpu_module(surf, K, float_type, n_workers_per_block, n_c2e_block_rows),
        traversal_module = get_traversal_module(K),
        n_workers_per_block = n_workers_per_block,
        n_c2e_block_rows = n_c2e_block_rows
    )
