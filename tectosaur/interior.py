import numpy as np
from tectosaur.util.quadrature import gauss2d_tri
import tectosaur.util.gpu as gpu

import tectosaur.fmm.fmm as fmm
from tectosaur.ops.sparse_integral_op import interp_galerkin_mat
from tectosaur.farfield import farfield_pts_direct

#TODO:
#1) Write using just one order and no nearfield/farfield split
#2) Separate into nearfield and farfield that can have different quadrature orders
#3) Use a correction for the nearfield so that the farfield can just be an all-pairs nbody problem
#4) Use FMM for the farfield component

@profile
def interior_integral(obs_pts, obs_ns, mesh, input, K, nq_far, nq_near, params, float_type):
    far_quad = gauss2d_tri(nq_far)
    IGmat, quad_pts, quad_ns = interp_galerkin_mat(
        mesh[0][mesh[1]], far_quad
    )
    gpu_quad_pts = gpu.to_gpu(quad_pts, float_type)
    gpu_quad_ns = gpu.to_gpu(quad_ns, float_type)

    interp_v = IGmat.dot(input)


    mac = 3.0
    order = 100
    pts_per_cell = 300
    cfg = fmm.make_config(K, params, 1.1, mac, order, pts_per_cell, float_type)

    obs_tree = fmm.make_tree(obs_pts, cfg)
    src_tree = fmm.make_tree(quad_pts, cfg)

    obs_orig_idxs = obs_tree.orig_idxs
    src_orig_idxs = src_tree.orig_idxs

    obs_ns_tree = obs_ns[obs_orig_idxs].copy()
    src_ns_tree = quad_ns[src_orig_idxs].copy()
    fmm_obj = fmm.FMM(obs_tree, obs_ns_tree, src_tree, src_ns_tree, cfg)

    fmm.report_interactions(fmm_obj)
    evaluator = fmm.FMMEvaluator(fmm_obj)

    input_tree = interp_v.reshape((-1,3))[src_orig_idxs,:].reshape(-1)

    fmm_out = fmm.eval(evaluator, input_tree.copy()).reshape((-1, 3))

    to_orig = np.empty_like(fmm_out)
    to_orig[obs_orig_idxs,:] = fmm_out
    to_orig = to_orig.flatten()
    return to_orig

    nbody_result = farfield_pts_direct(
        K, obs_pts, obs_ns, gpu_quad_pts, gpu_quad_ns, interp_v, params, float_type
    )
    return nbody_result
