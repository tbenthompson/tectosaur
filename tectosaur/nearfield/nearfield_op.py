import scipy.sparse
import numpy as np

import tectosaur.mesh.find_near_adj as find_near_adj

from tectosaur.nearfield.limit import richardson_quad
from tectosaur.nearfield.table_lookup import coincident_table, adjacent_table
from tectosaur.nearfield.pairs_integrator import PairsIntegrator

from tectosaur.util.timer import Timer
import tectosaur.util.sparse as sparse
import tectosaur.util.gpu as gpu

def to_sparse_mat(entries, pairs, shape):
    bcoo = sparse.BCOOMatrix(pairs[:, 0], pairs[:, 1], entries.reshape((-1, 9, 9)), shape)
    bsr = bcoo.to_bsr()
    return bcoo

def build_nearfield(shape, *mats):
    return [to_sparse_mat(*m, shape) for m in mats]

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
            self.shape,
            (co_mat - co_mat_correction, co_indices),
            (ea_mat_rot - ea_mat_correction, ea[:,:2]),
            (va_mat_rot - va_mat_correction, va[:,:2]),
            (nearfield_mat - nearfield_correction, nearfield_pairs)
        )
        timer.report("Assemble matrix")
        self.mat_no_correction = build_nearfield(
            self.shape,
            (co_mat, co_indices),
            (ea_mat_rot, ea[:,:2]),
            (va_mat_rot, va[:,:2]),
            (nearfield_mat, nearfield_pairs),
        )
        timer.report("Assemble uncorrected matrix")

        self.gpu_mat = None

    def dot(self, v):
        return sum(arr.dot(v) for arr in self.mat)

    def nearfield_no_correction_dot(self, v):
        return sum(arr.dot(v) for arr in self.mat_no_correction)
