import numpy as np

from tectosaur.nearfield.nearfield_op import NearfieldIntegralOp, get_gpu_module
from tectosaur.util.quadrature import gauss4d_tri
from tectosaur.ops.dense_op import DenseOp
import tectosaur.util.gpu as gpu
import tectosaur.viennacl as viennacl

from tectosaur import float_type

def farfield_tris(kernel, params, pts, obs_tris, src_tris, n_q):
    integrator = getattr(get_gpu_module(), "farfield_tris" + kernel)
    q = gauss4d_tri(n_q, n_q)

    gpu_qx, gpu_qw = gpu.quad_to_gpu(q, float_type)
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
            gpu.gpu_queue, (n_items, src_tris.shape[0]), None,
            gpu_result.data,
            np.int32(q[0].shape[0]), gpu_qx.data, gpu_qw.data,
            gpu_pts.data,
            np.int32(n_items), gpu_obs_tris.data,
            np.int32(src_tris.shape[0]), gpu_src_tris.data,
            gpu_params.data
        )
        out[start_idx:end_idx] = gpu_result.get()

    call_size = 1024
    for I in gpu.intervals(n, call_size):
        call_integrator(*I)

    return out

class DenseIntegralOp(DenseOp):
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, params, pts, tris,
            use_tables = False, remove_sing = False):

        nearfield = NearfieldIntegralOp(
            eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, params, pts, tris,
            use_tables, remove_sing
        ).mat_no_correction.todense()

        farfield = farfield_tris(
            kernel, params, pts, tris, tris, nq_far
        ).reshape(nearfield.shape)

        self.mat = np.where(np.abs(nearfield) > 0, nearfield, farfield)
        self.shape = self.mat.shape
        self.gpu_mat = None

    def dot(self, v):
        if self.gpu_mat is None:
            self.gpu_mat = gpu.to_gpu(self.mat, np.float32)
        return np.squeeze(viennacl.prod(self.gpu_mat, v, np.float32))
