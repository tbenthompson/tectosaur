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

from okada import Okada
import solve

from tectosaur.util.logging import setup_root_logger
logger = setup_root_logger(__name__)

def get_fault_slip(pts, fault_tris):
    dof_pts = pts[fault_tris]
    x = dof_pts[:,:,0]
    z = dof_pts[:,:,2]
    mean_z = np.mean(z)
    slip = np.zeros((fault_tris.shape[0], 3, 3))
    slip[:,:,0] = (1 - np.abs(x)) * (1 - np.abs((z - mean_z) * 2.0))
    slip[:,:,1] = (1 - np.abs(x)) * (1 - np.abs((z - mean_z) * 2.0))
    slip[:,:,2] = (1 - np.abs(x)) * (1 - np.abs((z - mean_z) * 2.0))
    # slip[:,:,0] = np.exp(-(x ** 2 + ((z - mean_z) * 2.0) ** 2) * 8.0)
    return slip

def build_constraints(surface_tris, fault_tris, pts):
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    cs = continuity_constraints(surface_tris, fault_tris)
    slip = get_fault_slip(pts, fault_tris)

    # slip_pts = np.zeros(pts.shape[0])
    # # slip_pts[fault_tris] = np.log10(np.abs(slip[:,:,0]))
    # slip_pts[fault_tris] = slip[:,:,0]
    # plt.tricontourf(pts[:,0], pts[:,2], fault_tris, slip_pts)
    # plt.triplot(pts[:,0], pts[:,2], fault_tris)
    # dof_pts = pts[fault_tris]
    # plt.xlim([np.min(dof_pts[:,:,0]), np.max(dof_pts[:,:,0])])
    # plt.ylim([np.min(dof_pts[:,:,2]), np.max(dof_pts[:,:,2])])
    # plt.colorbar()
    # plt.show()


    cs.extend(all_bc_constraints(
        n_surf_tris, n_surf_tris + n_fault_tris, slip.flatten()
    ))
    # cs.extend(free_edge_constraints(surface_tris))

    return cs

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
    near_threshold = 2.0
    if any_nearfield(
            data, data.surf_tri_idxs,
            data.fault_tri_idxs, near_threshold
            ):
        raise Exception("nearfield interactions not allowed!")
    else:
        print('good. all interactions are farfield.')

    cs = build_constraints(data.surface_tris, data.fault_tris, data.all_mesh[0])
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
    # T_op_fault_to_surf2 = SparseIntegralOp(
    #     6, 2, 5, near_threshold,
    #     'elasticRT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
    #     data.float_type,
    #     farfield_op_type = TriToTriDirectFarfieldOp,
    #     obs_subset = data.surf_tri_idxs,
    #     src_subset = data.fault_tri_idxs,
    # )
    T_op_fault_to_surf2 = TriToTriDirectFarfieldOp(
        2, 'elasticRT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
        data.float_type, obs_subset = data.surf_tri_idxs,
        src_subset = data.fault_tri_idxs
    )

    slip = get_fault_slip(data.all_mesh[0], data.fault_tris).reshape(-1)
    A = T_op_fault_to_surf.dot(slip).reshape((-1,3,3))
    B = T_op_fault_to_surf2.dot(slip).reshape((-1,3,3))
    ratio = A / B
    print(ratio)

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
    obj = Okada(41, 10, top_depth = -1.75)
    soln, soln2 = obj.run(build_and_solve = build_and_solve_T)
    # okada_soln = obj.okada_exact()
    obj.xsec_plot([soln, soln2], okada_soln = None)

if __name__ == "__main__":
    main()
