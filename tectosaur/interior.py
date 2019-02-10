import attr
import numpy as np
from tectosaur.util.quadrature import gauss2d_tri
import tectosaur.util.gpu as gpu
import scipy.sparse

import tectosaur.fmm.fmm as fmm
from tectosaur.farfield import farfield_pts_direct
import tectosaur.mesh.find_near_adj as find_near_adj
from tectosaur.kernels import kernels

def interp_galerkin_mat(tri_pts, quad_rule):
    import tectosaur.util.geometry as geometry
    nt = tri_pts.shape[0]
    qx, qw = quad_rule
    nq = qx.shape[0]

    rows = np.tile(
        np.arange(nt * nq * 3).reshape((nt, nq, 3))[:,:,np.newaxis,:], (1,1,3,1)
    ).flatten()
    cols = np.tile(
        np.arange(nt * 9).reshape(nt,3,3)[:,np.newaxis,:,:], (1,nq,1,1)
    ).flatten()

    basis = geometry.linear_basis_tri_arr(qx)

    unscaled_normals = geometry.unscaled_normals(tri_pts)
    jacobians = geometry.jacobians(unscaled_normals)

    b_tiled = np.tile((qw[:,np.newaxis] * basis)[np.newaxis,:,:], (nt, 1, 1))
    J_tiled = np.tile(jacobians[:,np.newaxis,np.newaxis], (1, nq, 3))
    vals = np.tile((J_tiled * b_tiled)[:,:,:,np.newaxis], (1,1,1,3)).flatten()

    quad_pts = np.zeros((nt * nq, 3))
    for d in range(3):
        for b in range(3):
            quad_pts[:,d] += np.outer(basis[:,b], tri_pts[:,b,d]).T.flatten()

    scaled_normals = unscaled_normals / jacobians[:,np.newaxis]
    quad_ns = np.tile(scaled_normals[:,np.newaxis,:], (1, nq, 1)).reshape((-1, 3))

    return scipy.sparse.coo_matrix((vals, (rows, cols))), quad_pts, quad_ns

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

#@profile
def interior_integral(obs_pts, obs_ns, mesh, input, K, nq_far, nq_near,
        params, float_type, fmm_params = None):
    threshold = 0.01

    # near_pairs = find_near_adj.fast_find_nearfield.get_nearfield(
    #     obs_pts, np.zeros(obs_pts.shape[0]),
    #     *find_near_adj.get_tri_centroids_rs(*mesh),
    #     threshold, 50
    # )

    # 2) perform some higher order quadrature for those pairs
    # 3) add in the corrections to cancel the FMM for those pairs

    far_quad = gauss2d_tri(nq_far)
    IGmat, quad_pts, quad_ns = interp_galerkin_mat(
        mesh[0][mesh[1]], far_quad
    )
    interp_v = IGmat.dot(input)

    if fmm_params is None:
        far = farfield_pts_direct(K, obs_pts, obs_ns, quad_pts, quad_ns, interp_v, params, float_type)
    else:
        far = interior_fmm_farfield(
            K, params, obs_pts, obs_ns, quad_pts, quad_ns, interp_v,
            float_type, fmm_params
        )
    return far

class InteriorOp:
    def __init__(self, obs_pts, obs_ns, src_mesh, K_name, nq_far,
            params, float_type):

        self.threshold = 1.0
        pairs = find_near_adj.fast_find_nearfield.get_nearfield(
            obs_pts, np.zeros(obs_pts.shape[0]),
            *find_near_adj.get_tri_centroids_rs(*src_mesh),
            self.threshold, 50
        )

        split = find_near_adj.split_vertex_nearfield(
            pairs, obs_pts, src_mesh[0], src_mesh[1]
        )
        self.vertex_pairs, self.near_pairs = split

        self.farfield = TriToPtDirectFarfieldOp(
            obs_pts, obs_ns, src_mesh, K_name, nq_far,
            params, float_type
        )

    def dot(self, v):
        return self.farfield.dot(v)

#TODO: A lot of duplication with TriToTriDirectFarfieldOp
class TriToPtDirectFarfieldOp:
    def __init__(self, obs_pts, obs_ns, src_mesh, K_name, nq,
            params, float_type):

        self.shape = (obs_pts.shape[0] * 3, src_mesh[1].shape[0] * 9)
        self.dim = obs_pts.shape[1]
        self.tensor_dim = kernels[K_name].tensor_dim
        self.n_obs = obs_pts.shape[0]
        self.n_src = src_mesh[1].shape[0]

        in_size = self.n_src * self.dim * self.tensor_dim
        out_size = self.n_obs * self.tensor_dim
        self.gpu_in = gpu.empty_gpu(in_size, float_type)
        self.gpu_out = gpu.empty_gpu(out_size, float_type)

        self.q = gauss2d_tri(nq)

        self.gpu_obs_pts = gpu.to_gpu(obs_pts, float_type)
        self.gpu_obs_ns = gpu.to_gpu(obs_ns, float_type)
        self.gpu_src_pts = gpu.to_gpu(src_mesh[0], float_type)
        self.gpu_src_tris = gpu.to_gpu(src_mesh[1], np.int32)
        self.gpu_params = gpu.to_gpu(np.array(params), float_type)
        self.block_size = 128
        self.n_blocks = int(np.ceil(self.n_obs / self.block_size))

        self.module = gpu.load_gpu(
            'farfield_tris.cl',
            tmpl_args = dict(
                block_size = self.block_size,
                float_type = gpu.np_to_c_type(float_type),
                quad_pts = self.q[0],
                quad_wts = self.q[1]
            )
        )
        self.fnc = getattr(self.module, "farfield_tris_to_pts" + K_name)

    def dot(self, v):
        self.gpu_in[:] = v[:].astype(self.gpu_in.dtype)
        self.fnc(
            self.gpu_out, self.gpu_in,
            self.gpu_obs_pts, self.gpu_obs_ns,
            self.gpu_src_pts, self.gpu_src_tris,
            self.gpu_params,
            np.int32(self.n_obs), np.int32(self.n_src),
            grid = (self.n_blocks, 1, 1), block = (self.block_size, 1, 1)
        )
        return self.gpu_out.get()
