import numpy as np
from test_farfield import make_meshes
from tectosaur.ops.sparse_integral_op import SparseIntegralOp
from tectosaur.ops.sparse_farfield_op import \
        TriToTriDirectFarfieldOp, PtToPtDirectFarfieldOp, PtToPtFMMFarfieldOp

def regularized_tester(K, sep, continuity):
    n_m = 20
    full_K_name = f'elastic{K}3'
    full_RK_name = f'elasticR{K}3'
    m, surf1_idxs, surf2_idxs = make_meshes(n_m = n_m, sep = sep)
    ops = [
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
            (TriToTriDirectFarfieldOp, full_RK_name),
            # (PtToPtFMMFarfieldOp(150, 2.5, 5), full_K_name)
        ]
    ]

    dof_pts = m[0][m[1][surf2_idxs]]
    dof_pts[:,:,1] -= dof_pts[0,0,1]

    def gaussian(a, b, c, x):
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    dist = np.linalg.norm(dof_pts.reshape(-1,3), axis = 1)
    slip = np.zeros((dof_pts.shape[0] * 3, 3))
    for d in range(3):
        slip[:,d] = gaussian(0.1 * d, 0.0, 0.3, dist)


    slip_flat = slip.flatten()
    outs = [o.dot(slip_flat) for o in ops]

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

        for o in ops:
            plot_at_pts(surf1_idxs, np.log10(np.abs(o.reshape(-1,3,3)[:,:,1]) + 1e-40))

    if continuity:
        from tectosaur.constraint_builders import continuity_constraints, \
            free_edge_constraints
        from tectosaur.constraints import build_constraint_matrix
        cs = continuity_constraints(m[1][surf1_idxs], np.array([]))
        cs.extend(free_edge_constraints(m[1][surf1_idxs]))
        cm, c_rhs = build_constraint_matrix(cs, outs[0].shape[0])
        final_outs = [cm.T.dot(v) for v in outs]
    else:
        final_outs = outs

    for i in range(len(outs)):
        print(outs[0] / outs[i])
        np.testing.assert_almost_equal(outs[0], outs[i], 6)

def test_regularized_T_farfield():
    regularized_tester('T', 4.0, False)

def test_regularized_A_farfield():
    regularized_tester('A', 4.0, True)

def test_regularized_H_farfield():
    regularized_tester('H', 4.0, True)
