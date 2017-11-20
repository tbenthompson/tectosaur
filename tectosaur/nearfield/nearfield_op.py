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
    if entries.shape[0] == 0:
        entries = np.empty((0, 9, 9))
    else:
        entries = entries.reshape((-1, 9, 9))
    bcoo = sparse.BCOOMatrix(pairs[:, 0], pairs[:, 1], entries, shape)
    return bcoo

def build_nearfield(shape, *mats):
    return [to_sparse_mat(*m, shape) for m in mats]

def setup_co(obs_subset, src_subset):
    co_tris = np.intersect1d(obs_subset, src_subset)
    co_indices = []
    for t in co_tris:
        co_indices.append([
            np.where(obs_subset == t)[0][0],
            np.where(src_subset == t)[0][0]
        ])
    co_indices = np.array(co_indices)
    if co_indices.shape[0] == 0:
        co_indices = np.empty((0, 2), dtype = np.int)
    return co_tris, co_indices

class NearfieldIntegralOp:
    def __init__(self, pts, tris, obs_subset, src_subset,
            nq_vert_adjacent, nq_far, nq_near, near_threshold,
            kernel, params, float_type):

        n_obs_dofs = obs_subset.shape[0] * 9
        n_src_dofs = src_subset.shape[0] * 9
        self.shape = (n_obs_dofs, n_src_dofs)

        timer = Timer(tabs = 1)
        pairs_int = PairsIntegrator(kernel, params, float_type, nq_far, nq_near, pts, tris)
        timer.report('setup pairs integrator')

        co_tris, co_indices = setup_co(obs_subset, src_subset)
        co_mat = coincident_table(kernel, params, pts[tris[co_tris]], float_type)
        timer.report("Coincident")
        co_mat_correction = pairs_int.correction(co_indices, True)
        timer.report("Coincident correction")

        close_or_touch_pairs = find_near_adj.find_close_or_touching(
            pts, tris[obs_subset], pts, tris[src_subset], near_threshold
        )
        nearfield_pairs, va, ea = find_near_adj.split_adjacent_close(
            close_or_touch_pairs, tris
        )
        print(nearfield_pairs.shape, va.shape, ea.shape)
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

    def to_dense(self):
        return sum([mat.to_bsr().to_scipy().todense() for mat in self.mat])

    def no_correction_to_dense(self):
        return sum([mat.to_bsr().to_scipy().todense() for mat in self.mat_no_correction])
