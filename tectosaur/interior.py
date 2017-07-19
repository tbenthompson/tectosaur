import numpy as np
from tectosaur.util.quadrature import gauss2d_tri
import tectosaur.util.gpu as gpu

from tectosaur.ops.sparse_integral_op import interp_galerkin_mat
from tectosaur.farfield import farfield_pts_direct

#TODO:
#1) Write using just one order and no nearfield/farfield split
#2) Separate into nearfield and farfield that can have different quadrature orders
#3) Use a correction for the nearfield so that the farfield can just be an all-pairs nbody problem
#4) Use FMM for the farfield component

def interior_integral(obs_pts, obs_ns, mesh, input, K, nq_far, nq_near, params):
    float_type = np.float32

    far_quad = gauss2d_tri(nq_far)
    IGmat, quad_pts, quad_ns = interp_galerkin_mat(
        mesh[0][mesh[1]], far_quad
    )
    gpu_quad_pts = gpu.to_gpu(quad_pts, float_type)
    gpu_quad_ns = gpu.to_gpu(quad_ns, float_type)

    interp_v = IGmat.dot(input)
    nbody_result = farfield_pts_direct(
        K, obs_pts, obs_ns, gpu_quad_pts, gpu_quad_ns, interp_v, params
    )
    return nbody_result
