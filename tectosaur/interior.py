import attr
import numpy as np
from tectosaur.util.quadrature import gauss2d_tri
import tectosaur.util.gpu as gpu

import tectosaur.fmm.fmm as fmm
from tectosaur.ops.sparse_integral_op import interp_galerkin_mat
from tectosaur.farfield import farfield_pts_direct

#TODO:
#2) Separate into nearfield and farfield that can have different quadrature orders
#3) Use a correction for the nearfield so that the farfield can just be an all-pairs nbody problem

def direct_interior_farfield(K, params, obs_pts, obs_ns, src_pts, src_ns, v, float_type):
    return farfield_pts_direct(K, obs_pts, obs_ns, src_pts, src_ns, v, params, float_type)

@attr.s()
class FMMInteriorFarfield:
    order = attr.ib()
    mac = attr.ib()
    obs_pts_per_cell = attr.ib()
    src_pts_per_cell = attr.ib()

    @profile
    def __call__(self, kernel, params, obs_pts, obs_ns, src_pts, src_ns, v, float_type):
        cfg = fmm.make_config(kernel, params, 1.1, self.mac, self.order, float_type)

        obs_tree = fmm.make_tree(obs_pts, cfg, self.obs_pts_per_cell)
        src_tree = fmm.make_tree(src_pts, cfg, self.src_pts_per_cell)

        obs_orig_idxs = obs_tree.orig_idxs
        src_orig_idxs = src_tree.orig_idxs

        obs_ns_tree = obs_ns[obs_orig_idxs].copy()
        src_ns_tree = src_ns[src_orig_idxs].copy()
        fmm_obj = fmm.FMM(obs_tree, obs_ns_tree, src_tree, src_ns_tree, cfg)

        fmm.report_interactions(fmm_obj)
        evaluator = fmm.FMMEvaluator(fmm_obj)

        input_tree = v.reshape((-1,3))[src_orig_idxs,:].reshape(-1)

        fmm_out = fmm.eval(evaluator, input_tree.copy()).reshape((-1, 3))

        to_orig = np.empty_like(fmm_out)
        to_orig[obs_orig_idxs,:] = fmm_out
        to_orig = to_orig.flatten()
        return to_orig

def interior_integral(obs_pts, obs_ns, mesh, input, K, nq_far, nq_near,
        params, float_type, farfield_fnc):

    far_quad = gauss2d_tri(nq_far)
    IGmat, quad_pts, quad_ns = interp_galerkin_mat(
        mesh[0][mesh[1]], far_quad
    )
    interp_v = IGmat.dot(input)

    return farfield_fnc(K, params, obs_pts, obs_ns, quad_pts, quad_ns, interp_v, float_type)
