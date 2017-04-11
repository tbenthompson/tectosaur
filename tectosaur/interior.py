import numpy as np
from tectosaur.quadrature import gauss2d_tri
import tectosaur.util.gpu as gpu

from tectosaur.sparse_integral_op import interp_galerkin_mat, farfield_pts_wrapper

def interior_integral(obs_pts, obs_ns, mesh, input, K, nq_far, nq_near, sm, pr):
    float_type = np.float32

    far_quad = gauss2d_tri(nq_far)
    IGmat, quad_pts, quad_ns = interp_galerkin_mat(
        mesh[0][mesh[1]], far_quad
    )
    import ipdb; ipdb.set_trace()
    gpu_quad_pts = gpu.to_gpu(quad_pts.flatten(), float_type)
    gpu_quad_ns = gpu.to_gpu(quad_ns.flatten(), float_type)

    interp_v = IGmat.dot(input).flatten()
    nbody_result = farfield_pts_wrapper(
        K, obs_pts.shape[0], obs_pts, obs_ns,
        quad_pts.shape[0], gpu_quad_pts, gpu_quad_ns, interp_v, sm, pr
    )
    return nbody_result


    # far_quad = gauss2d_tri(nq_far)
    # # near_quad = gauss2d_tri(nq_near)


    # tmpl_args = {'float_type': gpu.np_to_c_type(float_type)}
    # module = gpu.load_gpu('interior_integrals.cl', tmpl_args = tmpl_args)
    # fnc = getattr(module, 'interior_integrals' + K)

    # n_pts = obs_pts.shape[0]

    # gpu_result = gpu.empty_gpu((n_pts, 3), float_type)
    # gpu_far_quad_x = gpu.to_gpu(far_quad[0], float_type)
    # gpu_far_quad_w = gpu.to_gpu(far_quad[1], float_type)
    # gpu_obs_pts = gpu.to_gpu(obs_pts, float_type)
    # gpu_obs_ns = gpu.to_gpu(obs_ns, float_type)
    # gpu_mesh_pts = gpu.to_gpu(mesh[0], float_type)
    # gpu_mesh_tris = gpu.to_gpu(mesh[1], float_type)
    # gpu_input = gpu.to_gpu(input, float_type)

    # fnc(
    #     gpu.gpu_queue, (n_pts,), None,
    #     gpu_result.data,
    #     np.int32(far_quad[0].shape[0]),
    #     gpu_far_quad_x.data, gpu_far_quad_w.data,
    #     gpu_obs_pts.data, gpu_obs_ns.data,
    #     gpu_mesh_pts.data, np.int32(mesh[1].shape[0]), gpu_mesh_tris.data,
    #     gpu_input.data, np.float32(sm), np.float32(pr)
    # )

    # result = gpu_result.get()
    # print(result)



    # out = np.zeros((obs_pts.shape[0], 3))
    # for i in range(obs_pts.shape[0]):
    #     pt = obs_pts[i, :]
    #     for j in range(mesh[1].shape[0]):
    #         tri_pts = mesh[0][mesh[1][j, :]]
    #         element_pt(linear_basis_tri(


#1) write everything in python
#    --
#    -- use triangular gauss rule
#) Separate into nearfield and farfield that can have different quadrature orders
#) Use a correction for the nearfield so that the farfield can just be an all-pairs nbody problem
#) Use FMM for the farfield component
