import numpy as np
import matplotlib.pyplot as plt
from test_farfield import make_meshes
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, RegularizedSparseIntegralOp
from tectosaur.ops.dense_integral_op import DenseIntegralOp, RegularizedDenseIntegralOp
from tectosaur.ops.sparse_farfield_op import \
        TriToTriDirectFarfieldOp, PtToPtDirectFarfieldOp, PtToPtFMMFarfieldOp
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.neg_op import MultOp
from tectosaur.ops.sum_op import SumOp
from tectosaur.nearfield.nearfield_op import any_nearfield
from tectosaur.util.test_decorators import kernel
from tectosaur.util.timer import Timer

def plot_fnc(m, surf1_idxs, surf2_idxs, x, outs):
    def plot_at_pts(idxs, f):
        pts_f = np.full(m[0].shape[0], np.nan)
        pts_f[m[1][idxs]] = f

        pts_f_not_nan = pts_f[np.logical_not(np.isnan(pts_f))]
        min_f = np.min(pts_f_not_nan)
        max_f = np.max(pts_f_not_nan)

        plt.figure()
        plt.tricontourf(
            m[0][:,0], m[0][:,2], m[1], pts_f,
            levels = np.linspace(min_f, max_f, 21),
            extend = 'both'
        )
        plt.colorbar()


    for d in range(3):
        plot_at_pts(surf2_idxs, x[:,d].reshape((-1,3)))
        for o in outs:
            plot_at_pts(surf1_idxs, o.reshape(-1,3,3)[:,:,d])
        plt.show()

def build_x_field(m, surf1_idxs, surf2_idxs):
    dof_pts = m[0][m[1][surf2_idxs]]
    dof_pts[:,:,1] -= dof_pts[0,0,1]

    def gaussian(a, b, c, x):
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    dist = np.linalg.norm(dof_pts.reshape(-1,3), axis = 1)
    x = np.zeros((dof_pts.shape[0] * 3, 3))
    for d in range(3):
        x[:,d] = gaussian(0.1 * (d + 1), 0.0, 0.3, dist)
    return x

def x_ones_field(m, surf1_idxs, surf2_idxs):
    x = np.ones(surf2_idxs.shape[0] * 9).reshape((-1,3,3))
    from tectosaur.constraint_builders import find_free_edges
    free_edges = find_free_edges(m[1])
    first_tri_idx = np.min(surf2_idxs)
    last_tri_idx = np.max(surf2_idxs)
    for tri_idx, edge_idx in free_edges:
        if tri_idx < first_tri_idx or tri_idx > last_tri_idx:
            continue
        for v in range(2):
            pt_idx = m[1][tri_idx, (edge_idx + v) % 3]
            tris_touching = np.where(m[1] == pt_idx)
            x[tris_touching[0] - first_tri_idx, tris_touching[1], :] = 0.0
    return x

def regularized_tester(K, sep, continuity, mass_op_factor = 0.0, which = None):
    if which is None:
        raise Exception('select some operators!')

    n_m = 30
    full_K_name = f'elastic{K}3'
    full_RK_name = f'elasticR{K}3'
    m, surf1_idxs, surf2_idxs = make_meshes(n_m = n_m, sep = sep)
    if sep == 0.0:
        surf2_idxs = surf1_idxs

    near_threshold = 2.0
    nq_near = 5
    nq_far = 2

    if any_nearfield(m[0], m[1], surf1_idxs, surf2_idxs, near_threshold):
        nearfield = True
    else:
        nearfield = False

    def sparse_unregularized(far_op, Kn):
        return SparseIntegralOp(
            6, nq_far, nq_near, near_threshold, Kn, [1.0, 0.25], m[0], m[1],
            np.float32, farfield_op_type = far_op, obs_subset = surf1_idxs,
            src_subset = surf2_idxs
        )

    ops = [
        sparse_unregularized(PtToPtDirectFarfieldOp, full_K_name)
    ]

    if 'pt_to_pt_fmm' in which:
        ops.append(sparse_unregularized(PtToPtFMMFarfieldOp(150, 2.5, 5), full_K_name))

    if 'tri_farfield_regularized' in which:
        def change_K_tri_tri(*args):
            args = list(args)
            args[1] = full_RK_name
            return TriToTriDirectFarfieldOp(*args)
        ops.append(sparse_unregularized(change_K_tri_tri, full_K_name))

    if 'dense_regularized' in which:
        ops.append(SumOp([
            RegularizedDenseIntegralOp(
                10, 10, 6, nq_far, nq_near, near_threshold, full_RK_name, full_RK_name,
                [1.0, 0.25], m[0], m[1], np.float32,
                obs_subset = surf1_idxs, src_subset = surf2_idxs
            ),
            MultOp(MassOp(3, m[0], m[1][surf1_idxs]), mass_op_factor)
        ]))

    # Sparse regularized nearfield, FMM farfield
    if 'sparse_regularized' in which:
        ops.append(SumOp([
            RegularizedSparseIntegralOp(
                10, 10, 6, nq_far, nq_near, near_threshold, full_RK_name, full_K_name,
                [1.0, 0.25], m[0], m[1],
                np.float32,
                TriToTriDirectFarfieldOp,
                # PtToPtFMMFarfieldOp(150, 2.5, 5),
                obs_subset = surf1_idxs, src_subset = surf2_idxs
            ),
            MultOp(MassOp(3, m[0], m[1][surf1_idxs]), mass_op_factor)
        ]))

    print('built ops')

    x = build_x_field(m, surf1_idxs, surf2_idxs)
    x_flat = x.flatten()
    outs = [o.dot(x_flat) for o in ops]

    if continuity:
        from tectosaur.constraint_builders import continuity_constraints, \
            free_edge_constraints
        from tectosaur.constraints import build_constraint_matrix
        cs = continuity_constraints(m[1][surf1_idxs], np.array([]))
        cs.extend(free_edge_constraints(m[1][surf1_idxs]))
        cm, c_rhs = build_constraint_matrix(cs, outs[0].shape[0])
        final_outs = [cm.T.dot(v) for v in outs]
        plot_outs = [cm.dot(v) for v in final_outs]
    else:
        plot_outs = outs
        final_outs = outs

    should_plot = False
    if should_plot:
        plot_fnc(m, surf1_idxs, surf2_idxs, x, plot_outs)

    for i in range(len(final_outs)):
        for j in range(i + 1, len(final_outs)):
            print(i,j,final_outs[i] / final_outs[j])
            np.testing.assert_almost_equal(final_outs[i], final_outs[j], 6)

def test_regularized_T_farfield():
    regularized_tester('T', 2.0, False, which = ['tri_farfield_regularized'])

def test_regularized_A_farfield():
    regularized_tester('A', 2.0, True, which = ['tri_farfield_regularized'])

def test_regularized_H_farfield():
    regularized_tester('H', 4.0, True, which = ['tri_farfield_regularized'])

def test_regularized_T_nearfield():
    regularized_tester(
        'T', 0.4, False,
        which = ['dense_regularized', 'sparse_regularized']
    )

def test_regularized_A_nearfield():
    regularized_tester(
        'A', 0.4, True,
        which = ['dense_regularized', 'sparse_regularized']
    )

def test_regularized_H_nearfield():
    regularized_tester(
        'H', 0.4, True,
        which = ['dense_regularized', 'sparse_regularized']
    )

def test_regularized_T_self():
    regularized_tester(
        'T', 0.0, False, -0.5,
        which = ['dense_regularized', 'sparse_regularized']
    )

def test_regularized_A_self():
    regularized_tester(
        'A', 0.0, True, 0.5,
        which = ['dense_regularized', 'sparse_regularized']
    )

def test_regularized_H_self():
    regularized_tester(
        'H', 0.0, True,
        which = ['dense_regularized', 'sparse_regularized']
    )

def test_benchmark_far_tris():
    n = 100
    m, surf1_idxs, surf2_idxs = make_meshes(n_m = n, sep = 4.0)

    op = TriToTriDirectFarfieldOp(
        2, 'elasticRH3', [1.0, 0.25], m[0], m[1], np.float32,
        surf1_idxs, surf2_idxs
    )
    x = build_x_field(m, surf1_idxs, surf2_idxs)
    x_flat = x.flatten()

    op.dot(x_flat)
    import tectosaur, logging
    tectosaur.logger.setLevel(logging.INFO)
    n = 1
    t = Timer()
    for j in range(n):
        op.dot(x_flat)
    t.report('dot')

    op2 = PtToPtDirectFarfieldOp(
        2, 'elasticH3', [1.0, 0.25], m[0], m[1], np.float32,
        surf1_idxs, surf2_idxs
    )
    op2.dot(x_flat)
    t = Timer()
    for j in range(n):
        op2.dot(x_flat)
    t.report('dot pts')

def test_fun():
    n_m = 10
    sep = 0.05
    n_m2 = 10
    full_K_name = f'elasticT3'
    full_RK_name = f'elasticRT3'
    m, surf1_idxs, surf2_idxs = make_meshes(n_m = n_m, sep = sep, n_m2 = n_m2)
    mass_op_factor = 0.0

    do_self = False
    if do_self:
        surf2_idxs = surf1_idxs
        mass_op_factor = -0.5

    op1 = SparseIntegralOp(
        6, 3, 5, 2.0, full_K_name, [1.0, 0.25], m[0], m[1], np.float32,
        TriToTriDirectFarfieldOp,
        obs_subset = surf1_idxs, src_subset = surf2_idxs
    )

    op2 = SumOp([
        RegularizedSparseIntegralOp(
            10, 10, 6, 3, 5, 2.0, full_RK_name, full_K_name,
            [1.0, 0.25], m[0], m[1],
            np.float32, TriToTriDirectFarfieldOp,
            obs_subset = surf1_idxs, src_subset = surf2_idxs
        ),
        MultOp(MassOp(3, m[0], m[1][surf1_idxs]), mass_op_factor)
    ])

    x = x_ones_field(m, surf1_idxs, surf2_idxs)
    # x = build_x_field(m, surf1_idxs, surf2_idxs)

    y1 = op1.dot(x.flatten())
    y2 = op2.dot(x.flatten())
    print(y1, y2)
    plot_fnc(m, surf1_idxs, surf2_idxs, x.reshape((-1,3)), [y1, y2])

if __name__ == "__main__":
    test_benchmark_far_tris()
