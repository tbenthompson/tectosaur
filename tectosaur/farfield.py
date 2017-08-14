import numpy as np

import tectosaur.util.gpu as gpu

block_size = 128
def get_gpu_config(float_type):
    return dict(
        block_size = block_size,
        float_type = gpu.np_to_c_type(float_type)
    )

def get_gpu_module(float_type):
    return gpu.load_gpu('farfield_direct.cl', tmpl_args = get_gpu_config(float_type))

def farfield_pts_direct(K, obs_pts, obs_ns, src_pts, src_ns, vec, params, float_type):
    gpu_farfield_fnc = getattr(get_gpu_module(float_type), "farfield_pts" + K)

    n_obs, dim = obs_pts.shape
    n_src = src_pts.shape[0]

    tensor_dim = int(vec.shape[0] / n_src)

    gpu_result = gpu.empty_gpu(n_obs * tensor_dim, float_type)
    gpu_obs_pts = gpu.to_gpu(obs_pts, float_type)
    gpu_obs_ns = gpu.to_gpu(obs_ns, float_type)
    gpu_src_pts = gpu.to_gpu(src_pts, float_type)
    gpu_src_ns = gpu.to_gpu(src_ns, float_type)
    gpu_vec = gpu.to_gpu(vec, float_type)
    gpu_params = gpu.to_gpu(np.array(params), float_type)

    n_blocks = int(np.ceil(n_obs / block_size))
    gpu_farfield_fnc(
        gpu_result, gpu_obs_pts, gpu_obs_ns, gpu_src_pts, gpu_src_ns,
        gpu_vec, gpu_params, np.int32(n_obs), np.int32(n_src),
        grid = (n_blocks, 1, 1),
        block = (block_size, 1, 1)
    )
    return gpu_result.get()

