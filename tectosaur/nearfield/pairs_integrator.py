import numpy as np

import tectosaur.nearfield.triangle_rules as triangle_rules
import tectosaur.util.gpu as gpu
from tectosaur.util.quadrature import gauss4d_tri

def pairs_func_name(check0):
    check0_label = 'N'
    if check0:
        check0_label = 'Z'
    return 'single_pairs' + check0_label

def get_gpu_config(kernel, float_type):
    return dict(
        block_size = 128,
        float_type = gpu.np_to_c_type(float_type),
        kernel_name = kernel
    )

def get_gpu_module(kernel, float_type):
    return gpu.load_gpu('nearfield/nearfield.cl', tmpl_args = get_gpu_config(kernel, float_type))

class PairsIntegrator:
    def __init__(self, kernel, params, float_type, nq_far, nq_near, pts, tris):
        self.float_type = float_type
        self.module = get_gpu_module(kernel, float_type)
        self.gpu_params = gpu.to_gpu(np.array(params), self.float_type)
        self.gpu_near_q = self.quad_to_gpu(gauss4d_tri(nq_near, nq_near))
        self.gpu_far_q = self.quad_to_gpu(gauss4d_tri(nq_far, nq_far))
        self.gpu_pts = gpu.to_gpu(pts, self.float_type)
        self.gpu_tris = gpu.to_gpu(tris, np.int32)

    def quad_to_gpu(self, q):
        return [gpu.to_gpu(arr, self.float_type) for arr in q]

    def get_gpu_fnc(self, check0):
        return getattr(self.module, pairs_func_name(check0) + '_new')

    def pairs_quad(self, integrator, q, pairs_list):
        gpu_pairs_list = gpu.to_gpu(pairs_list.copy(), np.int32)
        n = pairs_list.shape[0]

        if n == 0:
            return np.empty((0,3,3,3,3), dtype = self.float_type)

        call_size = 2 ** 17
        gpu_result = gpu.empty_gpu((n, 3, 3, 3, 3), self.float_type)

        def call_integrator(start_idx, end_idx):
            integrator(
                gpu_result, np.int32(q[0].shape[0]), q[0], q[1],
                self.gpu_pts, self.gpu_tris,
                gpu_pairs_list, np.int32(start_idx), self.gpu_params,
                grid = (end_idx - start_idx, 1, 1), block = (1, 1, 1)
            )

        for I in gpu.intervals(n, call_size):
            call_integrator(*I)
        return gpu_result.get()

    def correction(self, pairs_list, check0):
        return self.pairs_quad(self.get_gpu_fnc(check0), self.gpu_far_q, pairs_list)

    def nearfield(self, pairs_list):
        return self.pairs_quad(self.get_gpu_fnc(False), self.gpu_near_q, pairs_list)

    def vert_adj(self, nq, pairs_list):
        integrator = getattr(self.module, pairs_func_name(False) + '_vert_adj')
        if type(nq) is int:
            nq = (nq, nq, nq)
        q = triangle_rules.vertex_adj_quad(nq[0], nq[1], nq[2])
        gpu_q = self.quad_to_gpu(q)
        return self.pairs_quad(integrator, gpu_q, pairs_list)

