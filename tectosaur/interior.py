import attr
import numpy as np
from tectosaur.util.quadrature import gauss2d_tri
import tectosaur.util.gpu as gpu

import tectosaur.fmm.fmm as fmm
from tectosaur.ops.sparse_integral_op import interp_galerkin_mat
from tectosaur.farfield import farfield_pts_direct
import tectosaur.mesh.find_near_adj as find_near_adj

def interior_fmm_farfield(kernel, params, obs_pts, obs_ns, src_pts, src_ns,
    v, float_type,  fmm_params):

    order, mac, obs_pts_per_cell, src_pts_per_cell = fmm_params

    cfg = fmm.make_config(kernel, params, 1.1, mac, order, float_type)

    obs_tree = fmm.make_tree(obs_pts, cfg, obs_pts_per_cell)
    src_tree = fmm.make_tree(src_pts, cfg, src_pts_per_cell)

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

@profile
def interior_integral(obs_pts, obs_ns, mesh, input, K, nq_far, nq_near,
        params, float_type, fmm_params = None):
    threshold = 1.0

    near_pairs = find_near_adj.fast_find_nearfield.get_nearfield(
        obs_pts, np.zeros(obs_pts.shape[0]),
        *find_near_adj.get_tri_centroids_rs(*mesh),
        threshold, 50
    )

    # 2) perform some higher order quadrature for those pairs
    # 3) add in the corrections to cancel the FMM for those pairs

    far_quad = gauss2d_tri(nq_far)
    IGmat, quad_pts, quad_ns = interp_galerkin_mat(
        mesh[0][mesh[1]], far_quad
    )
    interp_v = IGmat.dot(input)

    if fmm_params is None:
        far = farfield_pts_direct(K, obs_pts, obs_ns, src_pts, src_ns, v, params, float_type)
    else:
        far = interior_fmm_farfield(
            K, params, obs_pts, obs_ns, quad_pts, quad_ns, interp_v,
            float_type, fmm_params
        )
    return far

