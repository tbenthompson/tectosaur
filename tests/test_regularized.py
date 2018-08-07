import numpy as np
from test_farfield import make_meshes
from tectosaur.ops.sparse_integral_op import SparseIntegralOp
from tectosaur.ops.sparse_farfield_op import TriToTriDirectFarfieldOp
from tectosaur.ops.sparse_farfield_op import PtToPtDirectFarfieldOp
from tectosaur.ops.mass_op import MassOp

def regularized_tester(K, sep, continuity):
    full_K_name = f'elastic{K}3'
    full_RK_name = f'elasticR{K}3'
    m, surf1_idxs, surf2_idxs = make_meshes(n_m = 8, sep = sep)
    op1, op2 = [
        # SparseIntegralOp(
        #     6, 2, 5, 2.0, K, [1.0, 0.25], m[0], m[1],
        #     np.float32, farfield_op_type = C, obs_subset = surf1_idxs,
        #     src_subset = surf2_idxs
        # ) for C, K in [
        C(
            2, Kn, [1.0, 0.25], m[0], m[1],
            np.float32,
            obs_subset = surf1_idxs,
            src_subset = surf2_idxs
        ) for C, Kn in [
            (PtToPtDirectFarfieldOp, full_K_name),
            (TriToTriDirectFarfieldOp, full_RK_name)
        ]
    ]
    mass_op = MassOp(3, m[0], m[1])

    dof_pts = m[0][m[1][surf2_idxs]]
    dof_pts[:,:,1] -= dof_pts[0,0,1]

    def gaussian(a, b, c, x):
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    dist = np.linalg.norm(dof_pts.reshape(-1,3), axis = 1)
    slip = np.zeros((dof_pts.shape[0] * 3, 3))
    for d in range(3):
        slip[:,d] = gaussian(0.1 * d, 0.0, 0.3, dist)


    slip_flat = slip.flatten()
    out1 = op1.dot(slip_flat)
    out2 = op2.dot(slip_flat)

    should_plot = False
    def plot_at_pts(idxs, f):
        import matplotlib.pyplot as plt
        pts_f = np.zeros(m[0].shape[0])
        pts_f[m[1][idxs]] = f
        plt.tricontourf(m[0][:,0], m[0][:,2], m[1], pts_f)
        plt.colorbar()
        plt.show()

    if should_plot:
        plot_at_pts(surf2_idxs, np.log10(np.abs(slip[:,0].reshape((-1,3))) + 1e-40))
        out2[np.isinf(out2)] = 1e-10
        out2[np.abs(out2) < 1e-30] = 1e-10

        plot_at_pts(surf1_idxs, np.log10(np.abs(out1.reshape(-1,3,3)[:,:,1]) + 1e-40))
        plot_at_pts(surf1_idxs, np.log10(np.abs(out2.reshape(-1,3,3)[:,:,1]) + 1e-40))

    if continuity:
        from tectosaur.constraint_builders import continuity_constraints, \
            free_edge_constraints
        from tectosaur.constraints import build_constraint_matrix
        cs = continuity_constraints(m[1][surf1_idxs], np.array([]))
        cs.extend(free_edge_constraints(m[1][surf1_idxs]))
        cm, c_rhs = build_constraint_matrix(cs, out2.shape[0])
        out1 = cm.T.dot(out1)
        out2 = cm.T.dot(out2)

    print(out1 / out2)

    np.testing.assert_almost_equal(out1, out2, 6)

def test_regularized_T_farfield():
    regularized_tester('T', 2.0, False)

def test_regularized_A_farfield():
    regularized_tester('A', 2.0, True)

def test_regularized_H_farfield():
    regularized_tester('H', 4.0, True)
