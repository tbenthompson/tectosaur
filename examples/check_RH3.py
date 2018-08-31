import sys
import numpy as np
import matplotlib.pyplot as plt

import tectosaur.mesh.find_near_adj as find_near_adj
from tectosaur.constraint_builders import continuity_constraints, \
    free_edge_constraints, all_bc_constraints
from tectosaur.ops.sparse_integral_op import SparseIntegralOp
from tectosaur.ops.sparse_farfield_op import PtToPtDirectFarfieldOp, \
        PtToPtFMMFarfieldOp, TriToTriDirectFarfieldOp
from tectosaur.ops.dense_integral_op import DenseIntegralOp
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.composite_op import CompositeOp
from tectosaur.ops.sum_op import SumOp
from tectosaur.util.timer import Timer

from okada import Okada, build_constraints, abs_fault_slip
import solve

from tectosaur.util.logging import setup_root_logger
logger = setup_root_logger(__name__)

def build_and_solve_T(data):
    allow_nearfield = True
    near_threshold = 2.0
    if not allow_nearfield:
        if any_nearfield(
                data.all_mesh[0], data.all_mesh[1],
                data.surf_tri_idxs, data.fault_tri_idxs,
                near_threshold
                ):
            raise Exception("nearfield interactions not allowed!")
        else:
            print('good. all interactions are farfield.')

    cs = build_constraints(
        data.surface_tris, data.fault_tris, data.all_mesh[0],
        abs_fault_slip
    )
    op_type = SparseIntegralOp
    # op_type = DenseIntegralOp

    T_op = SparseIntegralOp(
        6, 2, 5, near_threshold,
        'elasticT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
        data.float_type,
        farfield_op_type = PtToPtFMMFarfieldOp(150, 3.0, 450),
        obs_subset = data.surf_tri_idxs,
        src_subset = data.surf_tri_idxs,
    )
    mass_op = MassOp(3, data.all_mesh[0], data.all_mesh[1])

    T_op_fault_to_surf = SparseIntegralOp(
        6, 2, 5, near_threshold,
        'elasticT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
        data.float_type,
        farfield_op_type = PtToPtDirectFarfieldOp,
        obs_subset = data.surf_tri_idxs,
        src_subset = data.fault_tri_idxs,
    )

    def replace_K_name(*args):
        args = list(args)
        args[1] = 'elasticRT3'
        return TriToTriDirectFarfieldOp(*args)
        # args[1] = 'elasticT3'
        # return PtToPtDirectFarfieldOp(*args)
        # return TriToTriDirectFarfieldOp(*args)

    T_op_fault_to_surf2 = DenseIntegralOp(
        6, 2, 10, near_threshold,
        'elasticRT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
        data.float_type,
        obs_subset = data.surf_tri_idxs,
        src_subset = data.fault_tri_idxs,
    )
    # T_op_fault_to_surf2 = SparseIntegralOp(
    #     6, 2, 5, near_threshold,
    #     'elasticRT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
    #     data.float_type,
    #     farfield_op_type = replace_K_name,
    #     obs_subset = data.surf_tri_idxs,
    #     src_subset = data.fault_tri_idxs,
    # )
    # T_op_fault_to_surf2 = TriToTriDirectFarfieldOp(
    #     2, 'elasticRT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
    #     data.float_type, obs_subset = data.surf_tri_idxs,
    #     src_subset = data.fault_tri_idxs
    # )

    slip = get_fault_slip(data.all_mesh[0], data.fault_tris).reshape(-1)
    A = T_op_fault_to_surf.dot(slip).reshape((-1,3,3))
    B = T_op_fault_to_surf2.dot(slip).reshape((-1,3,3))
    ratio = A / B
    import ipdb
    ipdb.set_trace()

    iop = CompositeOp(
        (mass_op, 0, 0),
        (T_op, 0, 0),
        (T_op_fault_to_surf, 0, data.n_surf_dofs),
        shape = (data.n_dofs, data.n_dofs)
    )
    iop2 = CompositeOp(
        (mass_op, 0, 0),
        (T_op, 0, 0),
        (T_op_fault_to_surf2, 0, data.n_surf_dofs),
        shape = (data.n_dofs, data.n_dofs)
    )

    return (
        solve.iterative_solve(iop, cs, tol = 1e-6),
        solve.iterative_solve(iop2, cs, tol = 1e-6)
    )

def main():
    obj = Okada(21, 10, top_depth = -0.2)
    soln, soln2 = obj.run(build_and_solve = build_and_solve_T)
    # okada_soln = obj.okada_exact()
    obj.xsec_plot([soln, soln2], okada_soln = None)

if __name__ == "__main__":
    main()
