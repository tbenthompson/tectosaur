import scipy.sparse
import numpy as np

import tectosaur.mesh.find_near_adj as find_near_adj

from tectosaur.nearfield.limit import richardson_quad
from tectosaur.nearfield.table_lookup import coincident_table, adjacent_table
from tectosaur.nearfield.pairs_integrator import PairsIntegrator

from tectosaur.util.timer import Timer
import tectosaur.util.gpu as gpu

from cppimport import cppimport
fast_assembly = cppimport("tectosaur.ops.fast_assembly")

def to_sparse_mat(entries, pairs):
    return entries.reshape((-1, 9, 9)), pairs[:, 0], pairs[:, 1]

def build_nearfield(co_data, ea_data, va_data, near_data, shape):
    t = Timer(tabs = 2)
    co_vals,co_rows,co_cols = to_sparse_mat(*co_data)
    ea_vals,ea_rows,ea_cols = to_sparse_mat(*ea_data)
    va_vals,va_rows,va_cols = to_sparse_mat(*va_data)
    near_vals,near_rows,near_cols = to_sparse_mat(*near_data)
    t.report("build pairs")

    #TODO: Pass each individual array to make_bsr_matrix and I can reduce
    # the copies by one.
    rows = np.concatenate((co_rows, ea_rows, va_rows, near_rows))
    cols = np.concatenate((co_cols, ea_cols, va_cols, near_cols))
    vals = np.concatenate((co_vals, ea_vals, va_vals, near_vals))
    t.report("stack pairs")

    #TODO: Make BSR could be done in-place with no memory allocation?
    data, indices, indptr = fast_assembly.make_bsr_matrix(
        shape[0], shape[1], 9, 9, vals, rows, cols
    )
    t.report("to bsr")
    mat = scipy.sparse.bsr_matrix((data, indices, indptr))
    t.report('make bsr')

    return mat

class NearfieldIntegralOp:
    @profile
    def __init__(self, nq_vert_adjacent, nq_far, nq_near,
            near_threshold, kernel, params, pts, tris, float_type):

        n = tris.shape[0] * 9
        self.shape = (n, n)

        timer = Timer(tabs = 1)
        pairs_int = PairsIntegrator(kernel, params, float_type, nq_far, nq_near, pts, tris)
        timer.report('setup pairs integrator')

        co_mat = coincident_table(kernel, params, pts, tris, float_type)
        timer.report("Coincident")
        co_indices = np.vstack([np.arange(tris.shape[0]) for i in [0,1]]).T
        co_mat_correction = pairs_int.correction(co_indices, True)
        timer.report("Coincident correction")

        close_or_touch_pairs = find_near_adj.find_close_or_touching(pts, tris, near_threshold)
        nearfield_pairs, va, ea = find_near_adj.split_adjacent_close(close_or_touch_pairs, tris)
        timer.report("Find nearfield/adjacency")

        ea_mat_rot = adjacent_table(nq_vert_adjacent, kernel, params, pts, tris, ea, float_type)
        timer.report("Edge adjacent")
        ea_mat_correction = pairs_int.correction(ea, False)
        timer.report("Edge adjacent correction")

        va_mat_rot = pairs_int.vert_adj(nq_vert_adjacent, va)
        timer.report("Vert adjacent")
        va_mat_correction = pairs_int.correction(va[:,:2], False)
        timer.report("Vert adjacent correction")

        nearfield_mat = pairs_int.nearfield(nearfield_pairs)
        timer.report("Nearfield")
        nearfield_correction = pairs_int.correction(nearfield_pairs, False)
        timer.report("Nearfield correction")

        self.mat = build_nearfield(
            (co_mat - co_mat_correction, co_indices),
            (ea_mat_rot - ea_mat_correction, ea[:,:2]),
            (va_mat_rot - va_mat_correction, va[:,:2]),
            (nearfield_mat - nearfield_correction, nearfield_pairs),
            self.shape
        )
        timer.report("Assemble matrix")
        self.mat_no_correction = build_nearfield(
            (co_mat, co_indices),
            (ea_mat_rot, ea[:,:2]),
            (va_mat_rot, va[:,:2]),
            (nearfield_mat, nearfield_pairs),
            self.shape
        )
        #TODO: Convert to using the base matrix and a correction matrix instead of "uncorrected"
        timer.report("Assemble uncorrected matrix")

        self.gpu_mat = None

    def dot(self, v):
        # if gpu.cuda_backend:
        #     from tectosaur.util.cusparse_bsr import cusparseBSR
        #     if self.gpu_mat is None:
        #         self.gpu_mat = cusparseBSR(self.mat)
        #     return self.gpu_mat.dot(v)
        # else:
        return self.mat.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.mat_no_correction.dot(v)
