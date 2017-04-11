

class FMMIntegralOp:
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, sm, pr, pts, tris):
        self.nearfield = NearfieldIntegralOp(
            eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, sm, pr, pts, tris
        )

        far_quad2d = gauss2d_tri(nq_far)
        self.interp_galerkin_mat, quad_pts, quad_ns = \
            interp_galerkin_mat(pts[tris], far_quad2d)
        self.shape = self.nearfield.mat.shape
        quad_ns = quad_ns.reshape(quad_pts.shape)

        order = 100
        mac = 3.0
        self.obs_kd = fmm.KDTree(quad_pts, quad_ns, order)
        self.src_kd = fmm.KDTree(quad_pts, quad_ns, order)
        self.fmm_mat = fmm.fmmmmmmm(
            self.obs_kd, self.src_kd,
            fmm.FMMConfig(1.1, mac, order, 'elastic' + kernel, [sm, pr])
        )

    def dot(self, v):
        out = self.nearfield.dot(v)
        out += self.farfield_dot(v)
        return out

    def farfield_dot(self, v):
        interp_v = self.interp_galerkin_mat.dot(v)
        fmm_out = fmm.eval(
            self.obs_kd, self.src_kd, self.fmm_mat, interp_v,
            interp_v.shape[0]
        )
        return self.interp_galerkin_mat.T.dot(fmm_out)

