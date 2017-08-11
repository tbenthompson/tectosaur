import attr
import numpy as np

import tectosaur.util.gpu as gpu

@attr.s
class Ball:
    center = attr.ib()
    R = attr.ib()

def inscribe_surf(ball, scaling, surf):
    return surf * ball.R * scaling + ball.center

def c2e_solve(gpu_module, surf, bounds, check_r, equiv_r, K, params, float_type):
    equiv_surf = inscribe_surf(bounds, equiv_r, surf)
    check_surf = inscribe_surf(bounds, check_r, surf)

    equiv_to_check = direct_matrix(
        gpu_module, K, check_surf, surf, equiv_surf, surf, params, float_type
    )

    rcond = np.finfo(float_type).eps * 30
    out = np.linalg.pinv(equiv_to_check, rcond = rcond).flatten()
    return out

def direct_matrix(gpu_module, K, obs_pts, obs_ns, src_pts, src_ns, params, float_type):
    gpu_obs_pts = gpu.to_gpu(obs_pts, float_type)
    gpu_obs_ns = gpu.to_gpu(obs_ns, float_type)
    gpu_src_pts = gpu.to_gpu(src_pts, float_type)
    gpu_src_ns = gpu.to_gpu(src_ns, float_type)
    gpu_params = gpu.to_gpu(params, float_type)

    n_obs = obs_pts.shape[0]
    n_src = src_pts.shape[0]
    out_shape = (n_obs, n_src)
    gpu_out = gpu.empty_gpu((n_obs * K.tensor_dim, n_src * K.tensor_dim), float_type)
    gpu_module.direct_matrix(
        gpu.gpu_queue, out_shape, None,
        gpu_out.data, gpu_obs_pts.data, gpu_obs_ns.data,
        gpu_src_pts.data, gpu_src_ns.data,
        np.int32(n_obs), np.int32(n_src), gpu_params.data
    )
    return gpu_out.get()

def build_c2e(tree, check_r, equiv_r, cfg):
    def make(R):
        return c2e_solve(
            cfg.gpu_module, cfg.surf,
            Ball([0] * cfg.K.spatial_dim, R), check_r, equiv_r,
            cfg.K, cfg.params, cfg.float_type
        )

    n_rows = cfg.K.tensor_dim * cfg.surf.shape[0]
    levels_to_compute = tree.max_height + 1
    if type(cfg.K.scale_type) is int:
        return make(1.0)

    c2e_ops = np.empty(levels_to_compute * n_rows * n_rows)
    for i in range(levels_to_compute):
        start_idx = i * n_rows * n_rows
        end_idx = (i + 1) * n_rows * n_rows
        c2e_ops[start_idx:end_idx] = make(tree.root().bounds.R / (2.0 ** i))

    return c2e_ops
