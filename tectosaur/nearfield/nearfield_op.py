import scipy.sparse
import numpy as np

import tectosaur.mesh.find_near_adj as find_near_adj

from tectosaur.nearfield.limit import richardson_quad
from tectosaur.nearfield.table_lookup import coincident_table, adjacent_table
from tectosaur.nearfield.pairs_integrator import PairsIntegrator

from tectosaur.util.timer import Timer
import tectosaur.util.sparse as sparse
import tectosaur.util.gpu as gpu

def to_dof_space(tri_indices, obs_subset, src_subset):
    dof_space_indices = []
    for pair in tri_indices:
        try:
            dof_space_indices.append([
                np.where(obs_subset == pair[0])[0][0],
                np.where(src_subset == pair[1])[0][0]
            ])
        except:
            import ipdb
            ipdb.set_trace()
    dof_space_indices = np.array(dof_space_indices)
    if dof_space_indices.shape[0] == 0:
        dof_space_indices = np.empty((0, 2), dtype = np.int)
    return dof_space_indices

def to_tri_space(dof_indices, obs_subset, src_subset):
    tri_idxs = np.array([obs_subset[dof_indices[:,0]], src_subset[dof_indices[:,1]]]).T
    return np.concatenate((tri_idxs, dof_indices[:,2:]), axis = 1)

def build_nearfield(shape, *mats):
    out = []
    for entries, pairs in mats:
        if entries.shape[0] == 0:
            entries = np.empty((0, 9, 9))
        else:
            entries = entries.reshape((-1, 9, 9))
        bcoo = sparse.BCOOMatrix(pairs[:, 0], pairs[:, 1], entries, shape)
        out.append(bcoo)
    return out

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

        co_tris = np.intersect1d(obs_subset, src_subset)
        co_indices = np.array([co_tris, co_tris]).T.copy()
        co_dofs = to_dof_space(co_indices, obs_subset, src_subset)

        co_mat = coincident_table(kernel, params, pts[tris[co_tris]], float_type)
        timer.report("Coincident")
        co_mat_correction = pairs_int.correction(co_indices, True)
        timer.report("Coincident correction")

        close_or_touch_pairs = find_near_adj.find_close_or_touching(
            pts, tris[obs_subset], pts, tris[src_subset], near_threshold
        )
        nearfield_pairs_dofs, va_dofs, ea_dofs = find_near_adj.split_adjacent_close(
            close_or_touch_pairs, tris[obs_subset], tris[src_subset]
        )
        nearfield_pairs = to_tri_space(nearfield_pairs_dofs, obs_subset, src_subset)
        va = to_tri_space(va_dofs, obs_subset, src_subset)
        ea = to_tri_space(ea_dofs, obs_subset, src_subset)
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
            (co_mat - co_mat_correction, co_dofs),
            (ea_mat_rot - ea_mat_correction, ea_dofs[:,:2]),
            (va_mat_rot - va_mat_correction, va_dofs[:,:2]),
            (nearfield_mat - nearfield_correction, nearfield_pairs_dofs)
        )
        timer.report("Assemble matrix")
        self.mat_no_correction = build_nearfield(
            self.shape,
            (co_mat, co_dofs),
            (ea_mat_rot, ea_dofs[:,:2]),
            (va_mat_rot, va_dofs[:,:2]),
            (nearfield_mat, nearfield_pairs_dofs),
        )
        timer.report("Assemble uncorrected matrix")

    def dot(self, v):
        return sum(arr.dot(v) for arr in self.mat)

    def nearfield_no_correction_dot(self, v):
        return sum(arr.dot(v) for arr in self.mat_no_correction)

    def to_dense(self):
        return sum([mat.to_bsr().to_scipy().todense() for mat in self.mat])

    def no_correction_to_dense(self):
        return sum([mat.to_bsr().to_scipy().todense() for mat in self.mat_no_correction])
