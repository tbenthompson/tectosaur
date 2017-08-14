import numpy as np

from tectosaur.nearfield.nearfield_op import NearfieldIntegralOp, get_gpu_module
from tectosaur.util.quadrature import gauss4d_tri
from tectosaur.ops.dense_op import DenseOp
import tectosaur.util.gpu as gpu

def farfield_tris(kernel, params, pts, obs_tris, src_tris, n_q, float_type):
    integrator = getattr(get_gpu_module(kernel, float_type), "farfield_tris")
    q = gauss4d_tri(n_q, n_q)

    gpu_qx = gpu.to_gpu(q[0], float_type)
    gpu_qw = gpu.to_gpu(q[1], float_type)
    gpu_pts = gpu.to_gpu(pts, float_type)
    gpu_src_tris = gpu.to_gpu(src_tris, np.int32)
    gpu_params = gpu.to_gpu(np.array(params), float_type)

    n = obs_tris.shape[0]
    out = np.empty(
        (n, 3, 3, src_tris.shape[0], 3, 3), dtype = float_type
    )

    def call_integrator(start_idx, end_idx):
        n_items = end_idx - start_idx
        gpu_result = gpu.empty_gpu((n_items, 3, 3, src_tris.shape[0], 3, 3), float_type)
        gpu_obs_tris = gpu.to_gpu(obs_tris[start_idx:end_idx], np.int32)
        integrator(
            gpu_result, np.int32(q[0].shape[0]), gpu_qx, gpu_qw,
            gpu_pts, np.int32(n_items), gpu_obs_tris,
            np.int32(src_tris.shape[0]), gpu_src_tris,
            gpu_params,
            grid = (n_items, src_tris.shape[0], 1),
            block = (1, 1, 1)
        )
        out[start_idx:end_idx] = gpu_result.get()

    call_size = 1024
    for I in gpu.intervals(n, call_size):
        call_integrator(*I)

    return out

class DenseIntegralOp(DenseOp):
    def __init__(self, nq_vert_adjacent, nq_far, nq_near,
            near_threshold, kernel, params, pts, tris, float_type):

        nearfield = NearfieldIntegralOp(
            nq_vert_adjacent, nq_far, nq_near,
            near_threshold, kernel, params, pts, tris, float_type
        ).mat_no_correction.todense()

        farfield = farfield_tris(
            kernel, params, pts, tris, tris, nq_far, float_type
        ).reshape(nearfield.shape)

        self.mat = np.where(np.abs(nearfield) > 0, nearfield, farfield)
        self.shape = self.mat.shape
        self.gpu_mat = None

    def dot(self, v):
        return self.mat.dot(v)
        # TODO: Use skcuda if available.
        # if self.gpu_mat is None:
        #     self.gpu_mat = gpu.to_gpu(self.mat, np.float32)
        # return np.squeeze(vcl.prod(self.gpu_mat, v, np.float32).get())
