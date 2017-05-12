import tectosaur.fmm_wrapper as fmm
from tectosaur.quadrature import gauss2d_tri

import numpy as np
from tectosaur.sparse_integral_op import NearfieldIntegralOp, interp_galerkin_mat, farfield_pts_wrapper
import tectosaur.util.gpu as gpu

class FMMIntegralOp:
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, sm, pr, pts, tris,
            use_tables = False, remove_sing = False):
        self.nearfield = NearfieldIntegralOp(
            eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, sm, pr, pts, tris,
            use_tables, remove_sing
        )

        far_quad2d = gauss2d_tri(nq_far)
        self.interp_galerkin_mat, quad_pts, quad_ns = \
            interp_galerkin_mat(pts[tris], far_quad2d)
        self.shape = self.nearfield.mat.shape
        quad_ns = quad_ns.reshape(quad_pts.shape)

        order = 200
        mac = 3.0
        self.obs_kd = fmm.KDTree(quad_pts, quad_ns, order)
        self.src_kd = fmm.KDTree(quad_pts, quad_ns, order)
        self.fmm_mat = fmm.fmmmmmmm(
            self.obs_kd, self.src_kd,
            fmm.FMMConfig(1.1, mac, order, 'elastic' + kernel, [sm, pr])
        )

        self.nq = quad_pts.shape[0]
        self.sm = sm
        self.pr = pr
        self.kernel = kernel
        self.gpu_quad_pts = gpu.to_gpu(quad_pts.flatten(), np.float32)
        self.gpu_quad_ns = gpu.to_gpu(quad_ns.flatten(), np.float32)

    def nearfield_dot(self, v):
        return self.nearfield.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.nearfield.mat_no_correction.dot(v)

    def dot(self, v):
        out = self.nearfield.dot(v)
        out += self.farfield_dot(v)
        return out

    def farfield_dot(self, v):
        interp_v = self.interp_galerkin_mat.dot(v)
        fmm_out = fmm.eval(self.fmm_mat, interp_v)
        nbody_result = farfield_pts_wrapper(
            self.kernel, self.nq, self.gpu_quad_pts, self.gpu_quad_ns,
            self.nq, self.gpu_quad_pts, self.gpu_quad_ns, interp_v, self.sm, self.pr
        )
        import ipdb; ipdb.set_trace()
        return self.interp_galerkin_mat.T.dot(fmm_out)

