import numpy as np
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray

from tectosaur.adjacency import find_adjacents, vert_adj_prep,\
    edge_adj_prep, rotate_tri
from tectosaur.find_nearfield import find_nearfield
from tectosaur.nearfield_op import coincident, pairs_quad, edge_adj, vert_adj,\
    get_gpu_module, get_gpu_config
from tectosaur.quadrature import gauss4d_tri
from tectosaur.util.timer import Timer

def farfield(kernel, sm, pr, pts, obs_tris, src_tris, n_q):
    q = gauss4d_tri(n_q, n_q)

    def call_integrator(block, grid, result_buf, tri_start, tri_end):
        integrator = get_gpu_module().get_function("farfield_tris" + kernel)
        if grid[0] == 0:
            return
        integrator(
            drv.Out(result_buf),
            np.int32(q[0].shape[0]),
            drv.In(q[0].astype(np.float32)),
            drv.In(q[1].astype(np.float32)),
            drv.In(pts.astype(np.float32)),
            np.int32(tri_end - tri_start),
            drv.In(obs_tris[tri_start:tri_end].astype(np.int32)),
            np.int32(src_tris.shape[0]),
            drv.In(src_tris.astype(np.int32)),
            np.float32(sm),
            np.float32(pr),
            block = block,
            grid = grid
        )

    max_block = 32
    max_grid = 20
    cur_result_buf = np.empty(
        (max_block * max_grid, 3, 3, src_tris.shape[0], 3, 3)
    ).astype(np.float32)

    full_result = np.empty(
        (obs_tris.shape[0], 3, 3, src_tris.shape[0], 3, 3)
    ).astype(np.float32)

    def next_integral(next_tri = 0):
        remaining = obs_tris.shape[0] - next_tri
        if remaining == 0:
            return
        elif remaining > max_block:
            block = (max_block, 1, 1)
        else:
            block = (remaining, 1, 1)
        grid = (min(remaining // block[0], max_grid), src_tris.shape[0])
        n_tris = grid[0] * block[0]
        past_end_tri = next_tri + n_tris
        call_integrator(block, grid, cur_result_buf, next_tri, past_end_tri)
        full_result[next_tri:past_end_tri] = cur_result_buf[:n_tris]
        next_integral(past_end_tri)

    next_integral()
    return full_result

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

def gpu_mvp(A, x):
    import skcuda.linalg as culg
    assert(A.dtype == np.float32)
    assert(x.dtype == np.float32)
    if type(A) != gpuarray.GPUArray:
        A = gpuarray.to_gpu(A)
    x_gpu = gpuarray.to_gpu(x)
    Ax_gpu = culg.dot(A, x_gpu)
    return Ax_gpu.get()

class DenseIntegralOp:
    def __init__(self, eps, nq_coincident, nq_edge_adjacent, nq_vert_adjacent,
            nq_far, nq_near, near_threshold, kernel, sm, pr, pts, tris):
        near_gauss = gauss4d_tri(nq_near, nq_near)

        timer = Timer(tabs = 1, silent = True)
        co_indices = np.arange(tris.shape[0])
        co_mat = coincident(nq_coincident, eps, kernel, sm, pr, pts, tris)
        timer.report("Coincident")

        va, ea = find_adjacents(tris)
        timer.report("Find adjacency")

        ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_obs_tris, ea_src_tris =\
            edge_adj_prep(tris, ea)
        timer.report("Edge adjacency prep")
        ea_mat_rot = edge_adj(
            nq_edge_adjacent, eps, kernel, sm, pr, pts, ea_obs_tris, ea_src_tris
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
            near_gauss, False
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
        import pycuda.gpuarray as gpuarray
        import skcuda.linalg as culg
        if self.gpu_mat is None:
            culg.init()
            self.gpu_mat = gpuarray.to_gpu(self.mat.astype(np.float32))
        return np.squeeze(gpu_mvp(self.gpu_mat, v[:,np.newaxis].astype(np.float32)))
