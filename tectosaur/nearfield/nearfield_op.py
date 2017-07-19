import scipy.sparse
import numpy as np

from tectosaur import float_type

from tectosaur.mesh.find_nearfield import find_nearfield
import tectosaur.mesh.adjacency as adjacency

from tectosaur.nearfield.limit import richardson_quad
from tectosaur.nearfield.table_lookup import coincident_table, adjacent_table
import tectosaur.nearfield.triangle_rules as triangle_rules

from tectosaur.util.quadrature import gauss4d_tri
from tectosaur.util.caching import cache
from tectosaur.util.timer import Timer
import tectosaur.util.gpu as gpu

from cppimport import cppimport
fast_assembly = cppimport("tectosaur.ops.fast_assembly")

def pairs_func_name(singular, k_name, check0):
    singular_label = 'N'
    if singular:
        singular_label = 'S'
    check0_label = 'N'
    if check0:
        check0_label = 'Z'
    return 'single_pairs' + singular_label + check0_label + k_name

def get_gpu_config():
    return {'block_size': 128, 'float_type': gpu.np_to_c_type(float_type)}

def get_gpu_module():
    return gpu.load_gpu('nearfield/nearfield.cl', tmpl_args = get_gpu_config())

def get_pairs_integrator(kernel, singular, check0):
    return getattr(get_gpu_module(), pairs_func_name(singular, kernel, check0))

def pairs_quad(kernel, params, pts, obs_tris, src_tris, q, singular, check0):
    integrator = get_pairs_integrator(kernel, singular, check0)

    gpu_qx, gpu_qw = gpu.quad_to_gpu(q, float_type)

    n = obs_tris.shape[0]
    out = np.empty((n, 3, 3, 3, 3), dtype = float_type)
    if n == 0:
        return out

    gpu_pts = gpu.to_gpu(pts, float_type)
    gpu_params = gpu.to_gpu(np.array(params), float_type)

    def call_integrator(start_idx, end_idx):
        n_items = end_idx - start_idx
        gpu_result = gpu.empty_gpu((n_items, 3, 3, 3, 3), float_type)
        gpu_obs_tris = gpu.to_gpu(obs_tris[start_idx:end_idx], np.int32)
        gpu_src_tris = gpu.to_gpu(src_tris[start_idx:end_idx], np.int32)
        integrator(
            gpu.gpu_queue, (n_items,), None,
            gpu_result.data,
            np.int32(q[0].shape[0]), gpu_qx.data, gpu_qw.data,
            gpu_pts.data, gpu_obs_tris.data, gpu_src_tris.data,
            gpu_params.data
        )
        out[start_idx:end_idx] = gpu_result.get()

    call_size = 2 ** 17
    for I in gpu.intervals(n, call_size):
        call_integrator(*I)
    return out

@cache
def cached_coincident_quad(nq, eps, remove_sing):
    if type(nq) is int:
        nq = (nq, nq, nq, nq)
    return richardson_quad(
        eps, remove_sing,
        lambda e: triangle_rules.coincident_quad(e, nq[0], nq[1], nq[2], nq[3])
    )

def coincident(nq, eps, kernel, params, pts, tris, remove_sing):
    q = cached_coincident_quad(nq, eps, remove_sing)
    out = pairs_quad(kernel, params, pts, tris, tris, q, True, True)
    return out

@cache
def cached_edge_adj_quad(nq, eps, remove_sing):
    if type(nq) is int:
        nq = (nq, nq, nq, nq)
    return richardson_quad(
        eps, remove_sing,
        lambda e: triangle_rules.edge_adj_quad(e, nq[0], nq[1], nq[2], nq[3], False)
    )

def edge_adj(nq, eps, kernel, params, pts, obs_tris, src_tris, remove_sing):
    q = cached_edge_adj_quad(nq, eps, remove_sing)
    out = pairs_quad(kernel, params, pts, obs_tris, src_tris, q, True, False)
    return out

@cache
def cached_vert_adj_quad(nq):
    if type(nq) is int:
        nq = (nq, nq, nq)
    return triangle_rules.vertex_adj_quad(nq[0], nq[1], nq[2])

def vert_adj(nq, kernel, params, pts, obs_tris, src_tris):
    q = cached_vert_adj_quad(nq)
    out = pairs_quad(kernel, params, pts, obs_tris, src_tris, q, False, False)
    return out

def pairs_sparse_mat(obs_idxs, src_idxs, integrals):
    return integrals.reshape((-1, 9, 9)), obs_idxs, src_idxs

def co_sparse_mat(co_indices, co_mat, correction):
    return pairs_sparse_mat(co_indices, co_indices, co_mat - correction)

def adj_sparse_mat(adj_mat, tri_idxs, obs_clicks, src_clicks, correction_mat):
    adj_mat = adj_mat.astype(np.float32)
    fast_assembly.derotate_adj_mat(adj_mat, obs_clicks, src_clicks)
    return pairs_sparse_mat(tri_idxs[:,0],tri_idxs[:,1], adj_mat - correction_mat)

def near_sparse_mat(near_mat, near_pairs, near_correction):
    return pairs_sparse_mat(near_pairs[:, 0], near_pairs[:, 1], near_mat - near_correction)

def build_nearfield(co_data, ea_data, va_data, near_data, shape):
    t = Timer(tabs = 2)
    co_vals,co_rows,co_cols = co_sparse_mat(*co_data)
    ea_vals,ea_rows,ea_cols = adj_sparse_mat(*ea_data)
    va_vals,va_rows,va_cols = adj_sparse_mat(*va_data)
    near_vals,near_rows,near_cols = near_sparse_mat(*near_data)
    t.report("build pairs")
    rows = np.concatenate((co_rows, ea_rows, va_rows, near_rows))
    cols = np.concatenate((co_cols, ea_cols, va_cols, near_cols))
    vals = np.concatenate((co_vals, ea_vals, va_vals, near_vals))
    t.report("stack pairs")

    data, indices, indptr = fast_assembly.make_bsr_matrix(
        shape[0], shape[1], 9, 9, vals, rows, cols
    )
    t.report("to bsr")
    mat = scipy.sparse.bsr_matrix((data, indices, indptr))
    t.report('make bsr')

    return mat


class NearfieldIntegralOp:
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, params, pts, tris,
            use_tables = False, remove_sing = False):
        n = tris.shape[0] * 9
        self.shape = (n, n)

        timer = Timer(tabs = 1)
        near_gauss = gauss4d_tri(nq_near, nq_near)
        far_quad = gauss4d_tri(nq_far, nq_far)
        timer.report("Build gauss rules")

        co_indices = np.arange(tris.shape[0])
        if not use_tables:
            co_mat = coincident(nq_coincident, eps, kernel, params, pts, tris, remove_sing)
        else:
            co_mat = coincident_table(kernel, params, pts, tris)
        timer.report("Coincident")
        co_mat_correction = pairs_quad(
            kernel, params, pts, tris, tris, far_quad, False, True
        )
        timer.report("Coincident correction")

        va, ea = adjacency.find_adjacents(tris)
        timer.report("Find adjacency")

        ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_obs_tris, ea_src_tris =\
            adjacency.edge_adj_prep(tris, ea)
        timer.report("Edge adjacency prep")
        if not use_tables:
            ea_mat_rot = edge_adj(
                nq_edge_adjacent, eps, kernel, params, pts,
                ea_obs_tris, ea_src_tris, remove_sing
            )
        else:
            ea_mat_rot = adjacent_table(
                nq_vert_adjacent, kernel, params, pts, ea_obs_tris, ea_src_tris
            )
        timer.report("Edge adjacent")
        ea_mat_correction = pairs_quad(
            kernel, params, pts,
            tris[ea_tri_indices[:,0]], tris[ea_tri_indices[:,1]],
            far_quad, False, False
        )
        timer.report("Edge adjacent correction")

        va_tri_indices, va_obs_clicks, va_src_clicks, va_obs_tris, va_src_tris =\
            adjacency.vert_adj_prep(tris, va)
        timer.report("Vert adjacency prep")

        va_mat_rot = vert_adj(
            nq_vert_adjacent, kernel, params, pts, va_obs_tris, va_src_tris
        )
        timer.report("Vert adjacent")
        va_mat_correction = pairs_quad(
            kernel, params, pts,
            tris[va_tri_indices[:,0]], tris[va_tri_indices[:,1]],
            far_quad, False, False
        )
        timer.report("Vert adjacent correction")

        nearfield_pairs = np.array(find_nearfield(pts, tris, va, ea, near_threshold))
        if nearfield_pairs.size == 0:
            nearfield_pairs = np.array([], dtype = np.int).reshape(0,2)
        timer.report("Find nearfield")

        nearfield_mat = pairs_quad(
            kernel, params, pts, tris[nearfield_pairs[:,0]], tris[nearfield_pairs[:, 1]],
            near_gauss, False, False
        )
        timer.report("Nearfield")
        nearfield_correction = pairs_quad(
            kernel, params, pts, tris[nearfield_pairs[:,0]], tris[nearfield_pairs[:, 1]],
            far_quad, False, False
        )
        timer.report("Nearfield correction")

        self.mat = build_nearfield(
            (co_indices, co_mat, co_mat_correction),
            (ea_mat_rot, ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_mat_correction),
            (va_mat_rot, va_tri_indices, va_obs_clicks, va_src_clicks, va_mat_correction),
            (nearfield_mat, nearfield_pairs, nearfield_correction),
            self.shape
        )
        timer.report("Assemble matrix")
        self.mat_no_correction = build_nearfield(
            (co_indices, co_mat, 0 * co_mat_correction),
            (ea_mat_rot, ea_tri_indices, ea_obs_clicks,
                ea_src_clicks, 0 * ea_mat_correction),
            (va_mat_rot, va_tri_indices, va_obs_clicks,
                va_src_clicks, 0 * va_mat_correction),
            (nearfield_mat, nearfield_pairs, 0 * nearfield_correction),
            self.shape
        )
        timer.report("Assemble uncorrected matrix")

    def dot(self, v):
        return self.mat.dot(v)
