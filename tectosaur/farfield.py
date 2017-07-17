import numpy as np

import tectosaur.util.gpu as gpu

from tectosaur import float_type

def get_gpu_config():
    return {'block_size': 128, 'float_type': gpu.np_to_c_type(float_type)}

def get_gpu_module():
    return gpu.load_gpu('farfield_direct.cl', tmpl_args = get_gpu_config())

def farfield_pts_direct(K, obs_pts, obs_ns, src_pts, src_ns, vec, params):
    dim = 3
    gpu_farfield_fnc = getattr(get_gpu_module(), "farfield_pts" + K + str(dim))

    n_obs = obs_pts.shape[0]
    if len(obs_pts.shape) == 1:
        n_obs //= dim
    n_src = src_pts.shape[0]
    if len(src_pts.shape) == 1:
        n_src //= dim

    tensor_dim = vec.shape[0] // n_obs

    gpu_result = gpu.empty_gpu(n_obs * tensor_dim, float_type)
    gpu_obs_pts = gpu.to_gpu(obs_pts, float_type)
    gpu_obs_ns = gpu.to_gpu(obs_ns, float_type)
    gpu_src_pts = gpu.to_gpu(src_pts, float_type)
    gpu_src_ns = gpu.to_gpu(src_ns, float_type)
    gpu_vec = gpu.to_gpu(vec, float_type)
    gpu_params = gpu.to_gpu(np.array(params), float_type)

    local_size = get_gpu_config()['block_size']
    n_blocks = int(np.ceil(n_obs / local_size))
    global_size = local_size * n_blocks
    ev = gpu_farfield_fnc(
        gpu.gpu_queue, (global_size,), (local_size,),
        gpu_result.data,
        gpu_obs_pts.data, gpu_obs_ns.data,
        gpu_src_pts.data, gpu_src_ns.data,
        gpu_vec.data,
        gpu_params.data,
        np.int32(n_obs), np.int32(n_src),
    )
    ev.wait()
    return gpu_result.get()

