import numpy as np
import matplotlib.pyplot as plt
from test_farfield import make_meshes
from tectosaur.ops.sparse_integral_op import SparseIntegralOp
from tectosaur.ops.dense_integral_op import RegularizedDenseIntegralOp
from tectosaur.ops.sparse_farfield_op import \
        TriToTriDirectFarfieldOp, PtToPtDirectFarfieldOp, PtToPtFMMFarfieldOp
from tectosaur.ops.mass_op import MassOp
from tectosaur.nearfield.nearfield_op import any_nearfield
from tectosaur.util.test_decorators import kernel

def plot_fnc(m, surf1_idxs, surf2_idxs, slip, outs):
    def plot_at_pts(idxs, f):
        pts_f = np.zeros(m[0].shape[0])
        pts_f[m[1][idxs]] = f
        plt.figure()
        plt.tricontourf(
            m[0][:,0], m[0][:,2], m[1], pts_f,
            # levels = np.linspace(-8, -4, 21),
            # extend = 'both'
        )
        plt.colorbar()

    # plot_at_pts(surf2_idxs, np.log10(np.abs(slip[:,0].reshape((-1,3))) + 1e-40))

    for o in outs:
        plot_at_pts(surf1_idxs, np.log10(np.abs(o.reshape(-1,3,3)[:,:,1]) + 1e-40))
    plt.show()



def regularized_tester(K, sep, continuity):
    n_m = 30
    full_K_name = f'elastic{K}3'
    full_RK_name = f'elasticR{K}3'
    m, surf1_idxs, surf2_idxs = make_meshes(n_m = n_m, sep = sep)
    if sep == 0.0:
        surf2_idxs = surf1_idxs

    near_threshold = 2.0
    if any_nearfield(m[0], m[1], surf1_idxs, surf2_idxs, near_threshold):
        nearfield = True
    else:
        nearfield = False

    sparse_ops = [
        (PtToPtDirectFarfieldOp, full_K_name),
    ]
    if not nearfield:
        def change_K_tri_tri(*args):
            args = list(args)
            args[1] = full_RK_name
            return TriToTriDirectFarfieldOp(*args)
        sparse_ops.extend([
            (PtToPtFMMFarfieldOp(150, 2.5, 5), full_K_name),
            (change_K_tri_tri, full_K_name)
        ])

    nq_near = 5
    ops = [
        SparseIntegralOp(
            6, 2, nq_near, 2.0, Kn, [1.0, 0.25], m[0], m[1],
            np.float32, farfield_op_type = C, obs_subset = surf1_idxs,
            src_subset = surf2_idxs
        ) for C, Kn in sparse_ops
    ]
    ops.append(RegularizedDenseIntegralOp(
        6, 2, nq_near, 2.0, full_RK_name, [1.0, 0.25], m[0], m[1],
        np.float32, obs_subset = surf1_idxs,
        src_subset = surf2_idxs
    ))

    dof_pts = m[0][m[1][surf2_idxs]]
    dof_pts[:,:,1] -= dof_pts[0,0,1]

    def gaussian(a, b, c, x):
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    dist = np.linalg.norm(dof_pts.reshape(-1,3), axis = 1)
    slip = np.zeros((dof_pts.shape[0] * 3, 3))
    for d in range(3):
        slip[:,d] = gaussian(0.1 * (d + 1), 0.0, 0.3, dist)


    slip_flat = slip.flatten()
    ops[1].dot(slip_flat)
    outs = [o.dot(slip_flat) for o in ops]

    # mass_op = MassOp(3, m[0], m[1][surf1_idxs])
    # outs[1] -= 0.5 * mass_op.dot(slip_flat)

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

    should_plot = True
    if should_plot:
        plot_fnc(m, surf1_idxs, surf2_idxs, slip, outs)

    for i in range(len(final_outs)):
        for j in range(i + 1, len(final_outs)):
            print(i,j,final_outs[i] / final_outs[j])
            np.testing.assert_almost_equal(final_outs[i], final_outs[j], 6)

def test_regularized_T_farfield():
    regularized_tester('T', 2.0, False)

def test_regularized_A_farfield():
    regularized_tester('A', 2.0, True)

def test_regularized_H_farfield():
    regularized_tester('H', 4.0, True)

def test_regularized_nearfield():
    nearfield_dist = 0.4
    K_char = 'A'
    if K_char == 'U':
        return
    regularized_tester(K_char, nearfield_dist, False)

def test_regularized_T_self():
    regularized_tester('T', 0.0, False)

def test_regularized_H_self():
    regularized_tester('H', 0.0, True)
