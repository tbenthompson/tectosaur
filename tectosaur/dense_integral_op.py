import numpy as np

from tectosaur.adjacency import find_adjacents, vert_adj_prep,\
    edge_adj_prep, rotate_tri
from tectosaur.find_nearfield import find_nearfield
from tectosaur.nearfield_op import coincident, pairs_quad, edge_adj, vert_adj,\
    get_gpu_module, float_type
from tectosaur.dense_op import DenseOp
from tectosaur.quadrature import gauss4d_tri
from tectosaur.util.timer import Timer
from tectosaur.table_lookup import coincident_table, adjacent_table
import tectosaur.util.gpu as gpu
import tectosaur.viennacl as viennacl

def farfield(kernel, sm, pr, pts, obs_tris, src_tris, n_q):
    integrator = getattr(get_gpu_module(), "farfield_tris" + kernel)
    q = gauss4d_tri(n_q, n_q)

    gpu_qx, gpu_qw = gpu.quad_to_gpu(q, float_type)
    gpu_pts = gpu.to_gpu(pts, float_type)
    gpu_src_tris = gpu.to_gpu(src_tris, np.int32)

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
            float_type(sm), float_type(pr)
        )
        out[start_idx:end_idx] = gpu_result.get()

    call_size = 1024
    for I in gpu.intervals(n, call_size):
        call_integrator(*I)

    return out

def set_co_entries(mat, co_mat, co_indices):
    mat[co_indices, :, :, co_indices, :, :] = co_mat
    return mat

def set_adj_entries(mat, adj_mat, tri_idxs, obs_clicks, src_clicks):
    obs_derot = rotate_tri(-obs_clicks)
    src_derot = rotate_tri(-src_clicks)

    placeholder = np.arange(adj_mat.shape[0])
    for b1 in range(3):
        for b2 in range(3):
            mat[tri_idxs[:,0], b1, :,tri_idxs[:,1], b2, :] = \
                adj_mat[placeholder, obs_derot[b1], :, src_derot[b2], :]
    return mat

def set_near_entries(mat, near_mat, near_entries):
    mat[near_entries[:,0],:,:,near_entries[:,1],:,:] = near_mat
    return mat

class DenseIntegralOp(DenseOp):
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, sm, pr, pts, tris,
            use_tables = False, remove_sing = False):
        near_gauss = gauss4d_tri(nq_near, nq_near)

        timer = Timer(tabs = 1, silent = True)
        co_indices = np.arange(tris.shape[0])
        if not use_tables:
            co_mat = coincident(nq_coincident, eps, kernel, sm, pr, pts, tris, remove_sing)
        else:
            co_mat = coincident_table(kernel, sm, pr, pts, tris)
        timer.report("Coincident")

        va, ea = find_adjacents(tris)
        timer.report("Find adjacency")

        ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_obs_tris, ea_src_tris =\
            edge_adj_prep(tris, ea)
        timer.report("Edge adjacency prep")
        if not use_tables:
            ea_mat_rot = edge_adj(
                nq_edge_adjacent, eps, kernel, sm, pr, pts, ea_obs_tris, ea_src_tris, remove_sing
            )
        else:
            ea_mat_rot = adjacent_table(
                nq_vert_adjacent, kernel, sm, pr, pts, ea_obs_tris, ea_src_tris
            )
        timer.report("Edge adjacent")

        va_tri_indices, va_obs_clicks, va_src_clicks, va_obs_tris, va_src_tris =\
            vert_adj_prep(tris, va)
        timer.report("Vert adjacency prep")
        va_mat_rot = vert_adj(nq_vert_adjacent, kernel, sm, pr, pts, va_obs_tris, va_src_tris)
        timer.report("Vert adjacent")

        nearfield_pairs = np.array(find_nearfield(pts, tris, va, ea, near_threshold))
        if nearfield_pairs.size == 0:
            nearfield_pairs = np.array([], dtype = np.int).reshape(0,2)
        nearfield_mat = pairs_quad(
            kernel, sm, pr, pts, tris[nearfield_pairs[:,0]], tris[nearfield_pairs[:, 1]],
            near_gauss, False, False
        )
        timer.report("Nearfield")

        out = farfield(kernel, sm, pr, pts, tris, tris, nq_far)

        timer.report("Farfield")
        out = set_co_entries(out, co_mat, co_indices)
        out = set_adj_entries(
            out, ea_mat_rot, ea_tri_indices, ea_obs_clicks, ea_src_clicks
        )
        out = set_adj_entries(
            out, va_mat_rot, va_tri_indices, va_obs_clicks, va_src_clicks
        )
        out = set_near_entries(out, nearfield_mat, nearfield_pairs)
        timer.report("Insert coincident adjacent")

        out.shape = (
            out.shape[0] * out.shape[1] * out.shape[2],
            out.shape[3] * out.shape[4] * out.shape[5]
        )

        self.mat = out
        self.shape = self.mat.shape
        self.gpu_mat = None

    def dot(self, v):
        if self.gpu_mat is None:
            self.gpu_mat = gpu.to_gpu(self.mat, np.float32)
        return np.squeeze(viennacl.prod(self.gpu_mat, v, np.float32))
