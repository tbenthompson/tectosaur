import sys
import numpy as np
import matplotlib.pyplot as plt

import tectosaur.mesh.find_near_adj as find_near_adj
from tectosaur.constraint_builders import continuity_constraints, \
    free_edge_constraints
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FarfieldOp
from tectosaur.ops.dense_integral_op import DenseIntegralOp
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.composite_op import CompositeOp
from tectosaur.ops.sum_op import SumOp
from tectosaur.util.timer import Timer

from okada import Okada, build_constraints
import solve

from tectosaur.util.logging import setup_root_logger
logger = setup_root_logger(__name__)

def any_nearfield(data, obs_subset, src_subset, near_threshold):
    pts, tris = data.all_mesh
    close_or_touch_pairs = find_near_adj.find_close_or_touching(
        pts, tris[obs_subset], pts, tris[src_subset], near_threshold
    )
    nearfield_pairs_dofs, va_dofs, ea_dofs = find_near_adj.split_adjacent_close(
        close_or_touch_pairs, tris[obs_subset], tris[src_subset]
    )
    return nearfield_pairs_dofs.shape[0] > 0

def build_and_solve_T(data):

    cs = build_constraints(data.surface_tris, data.fault_tris, data.all_mesh[0])
    op_type = SparseIntegralOp
    # op_type = DenseIntegralOp

    near_threshold = 2.0
    T_op = SparseIntegralOp(
        6, 2, 5, near_threshold,
        'elasticT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
        data.float_type,
        obs_subset = data.surf_tri_idxs,
        src_subset = data.surf_tri_idxs,
    )
    mass_op = MassOp(3, data.all_mesh[0], data.all_mesh[1])

    if any_nearfield(
            data, data.surf_tri_idxs,
            data.fault_tri_idxs, near_threshold
            ):
        raise Exception("nearfield interactions not allowed!")

    T_op_fault_to_surf = FarfieldOp(
        2, 'elasticT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
        data.float_type, obs_subset = data.surf_tri_idxs,
        src_subset = data.fault_tri_idxs
    )

    iop = CompositeOp(
        (mass_op, 0, 0),
        (T_op, 0, 0),
        (T_op_fault_to_surf, 0, data.n_surf_dofs),
        shape = (data.n_dofs, data.n_dofs)
    )

    return solve.iterative_solve(iop, cs, tol = 1e-6)

def main():
    obj = Okada(21, 3, top_depth = -2.7)
    soln = obj.run(build_and_solve = build_and_solve_T)
    okada_soln = obj.okada_exact()
    obj.xsec_plot(soln, okada_soln)

if __name__ == "__main__":
    main()
