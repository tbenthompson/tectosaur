import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.util.timer import Timer

import cppimport
_fmm = cppimport.imp("tectosaur._fmm._fmm")._fmm._fmm
for k in dir(_fmm):
    locals()[k] = getattr(_fmm, k)

float_type = np.float32
def gpu_p2p_eval(fmm_mat, input_vals):
    f = gpu.load_gpu('_fmm/p2p_kernel.cl', tmpl_args = dict()).p2p_kernel

    t = Timer()
    #TODO: Benchmark and check if its worth exposing the
    # buffer interface for these arrays to avoid copying the data
    gpu_obs_pts = gpu.to_gpu(np.array(fmm_mat.obs_tree.pts), float_type)
    gpu_obs_normals = gpu.to_gpu(np.array(fmm_mat.obs_tree.normals), float_type)
    gpu_src_pts = gpu.to_gpu(np.array(fmm_mat.src_tree.pts), float_type)
    gpu_src_normals = gpu.to_gpu(np.array(fmm_mat.src_tree.normals), float_type)

    gpu_obs_n_start = gpu.to_gpu(np.array(fmm_mat.p2p.obs_n_start), np.int32)
    gpu_obs_n_end = gpu.to_gpu(np.array(fmm_mat.p2p.obs_n_end), np.int32)
    gpu_src_n_start = gpu.to_gpu(np.array(fmm_mat.p2p.src_n_start), np.int32)
    gpu_src_n_end = gpu.to_gpu(np.array(fmm_mat.p2p.src_n_end), np.int32)

    gpu_out = gpu.zeros_gpu(gpu_obs_pts.shape[0], float_type)
    gpu_in = gpu.to_gpu(input_vals, float_type)
    t.report("data to gpu")

    f(
        gpu.gpu_queue, (gpu_obs_n_start.shape[0],), None,
        gpu_out.data, gpu_in.data,
        gpu_obs_n_start.data, gpu_obs_n_end.data,
        gpu_src_n_start.data, gpu_src_n_end.data,
        gpu_obs_pts.data, gpu_obs_normals.data,
        gpu_src_pts.data, gpu_src_normals.data
    )
    retval = gpu_out.get()
    t.report("run")
    return retval


def eval(fmm_mat, input_vals):
    # Need to call invidivual steps? How to substitute in the gpu
    t = Timer()
    out = fmm_mat.p2p_eval(input_vals)
    t.report("old p2p")
    for i in range(2):
        out = gpu_p2p_eval(fmm_mat, input_vals)
        t.report("new p2p")
    out += fmm_mat.eval(input_vals)
    return out
