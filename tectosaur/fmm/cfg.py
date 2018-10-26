import numpy as np
import attr

import tectosaur.util.gpu as gpu
from tectosaur.util.quadrature import gauss4d_tri
from tectosaur.kernels import kernels
from tectosaur.mesh.mesh_gen import make_sphere

from tectosaur.fmm.surrounding_surf import surrounding_surf

from tectosaur.util.cpp import imp
traversal_ext = imp("tectosaur.fmm.traversal_wrapper")

def get_dim_module(dim):
    return traversal_ext.two if dim == 2 else traversal_ext.three

def get_traversal_module(K):
    if type(K.scale_type) is int:
        return get_dim_module(K.spatial_dim).kdtree
    else:
        return get_dim_module(K.spatial_dim).octree

def get_gpu_module(surf, quad, K, float_type, n_workers_per_block, n_c2e_block_rows):
    args = dict(
        n_workers_per_block = n_workers_per_block,
        n_c2e_block_rows = n_c2e_block_rows,
        gpu_float_type = gpu.np_to_c_type(float_type),
        surf_pts = surf[0],
        surf_tris = surf[1],
        quad_pts = quad[0],
        quad_wts = quad[1],
        K = K
    )
    gpu_module = gpu.load_gpu(
        'fmm/tri_gpu_kernels.cl',
        tmpl_args = args
    )
    return gpu_module

@attr.s
class FMMConfig:
    K = attr.ib()
    params = attr.ib()
    surf = attr.ib()
    quad = attr.ib()
    outer_r = attr.ib()
    inner_r = attr.ib()
    alpha = attr.ib()
    float_type = attr.ib()
    gpu_module = attr.ib()
    traversal_module = attr.ib()
    n_workers_per_block = attr.ib()
    n_c2e_block_rows = attr.ib()
    treecode = attr.ib()
    order = attr.ib()

def make_config(K_name, params, inner_r, outer_r, order,
        float_type, alpha = 1e-5, n_workers_per_block = 64, n_c2e_block_rows = 16,
        treecode = False, force_order = None):

    K = kernels[K_name]
    quad = gauss4d_tri(2, 2)
    surf = make_sphere((0.0, 0.0, 0.0), 1.0, order)
    order = surf[1].shape[0]
    if force_order is not None:
        order = force_order
    if len(params) == 0:
        params = [0.0]
    return FMMConfig(
        K = K,
        params = np.array(params),
        surf = surf,
        quad = quad,
        outer_r = outer_r,
        inner_r = inner_r,
        alpha = alpha,
        float_type = float_type,
        gpu_module = get_gpu_module(surf, quad, K, float_type, n_workers_per_block, n_c2e_block_rows),
        traversal_module = get_traversal_module(K),
        n_workers_per_block = n_workers_per_block,
        n_c2e_block_rows = n_c2e_block_rows,
        treecode = treecode,
        order = order
    )
