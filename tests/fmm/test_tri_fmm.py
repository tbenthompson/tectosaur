import logging

import numpy as np
import matplotlib.pyplot as plt

import tectosaur as tct
from tectosaur.ops.sparse_farfield_op import (
    TriToTriDirectFarfieldOp, FMMFarfieldOp, PtToPtDirectFarfieldOp)
from tectosaur.ops.sparse_integral_op import RegularizedSparseIntegralOp
from tectosaur.ops.dense_integral_op import farfield_tris
import tectosaur.mesh.mesh_gen as mesh_gen
from tectosaur.kernels import kernels
from tectosaur.mesh.modify import concat
import tectosaur.util.gpu as gpu
from tectosaur.util.quadrature import gauss4d_tri
from tectosaur.ops.dense_integral_op import RegularizedDenseIntegralOp
from tectosaur.fmm.fmm import make_tree, make_config, FMM, FMMEvaluator, report_interactions
from tectosaur.fmm.c2e import reg_lstsq_inverse
from tectosaur.constraint_builders import continuity_constraints
from tectosaur.constraints import build_constraint_matrix
from tectosaur.util.timer import Timer
from tectosaur.util.test_decorators import golden_master

# TODO: dim
def test_tri_ones():
    from test_farfield import make_meshes
    m, surf1_idxs, surf2_idxs = make_meshes(n_m = 2, sep = 5, w = 1)
    ops = [
        C(
            3, 'tensor_one3', [], m[0], m[1], np.float64,
            surf1_idxs, surf2_idxs
        ) for C in [
            TriToTriDirectFarfieldOp,
            PtToPtDirectFarfieldOp,
            # PtToPtFMMFarfieldOp(250, 3.0, 250)
        ]
    ]

    x = np.random.rand(ops[0].shape[1])
    x = np.ones(ops[0].shape[1])
    ys = [op.dot(x) for op in ops]
    for y in ys:
        print(y)

def op(obs_m, src_m, K, nq = 2):
    new_pts, new_tris = concat(obs_m, src_m)
    n_obs_tris = obs_m[1].shape[0]
    obs_tris = new_tris[:n_obs_tris]
    src_tris = new_tris[n_obs_tris:]
    save = False
    if save:
        np.save('ptspts.npy', [new_pts, obs_tris, src_tris, nq])
    mat = farfield_tris(
        K, [1.0, 0.25], new_pts, obs_tris, src_tris, nq, np.float32
    )
    nrows = mat.shape[0] * 9
    ncols = mat.shape[3] * 9
    return mat.reshape((nrows, ncols))

def test_tri_fmm_p2p():
    np.random.seed(100)

    n = 10
    m1 = mesh_gen.make_rect(n, n, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    m2 = mesh_gen.make_rect(n, n, [[-3, 0, 1], [-3, 0, -1], [-2, 0, -1], [-2, 0, 1]])

    K = 'elasticRH3'
    cfg = make_config(K, [1.0, 0.25], 1.1, 2.5, 2, np.float32, treecode = True, force_order = 1000000)
    tree1 = make_tree(m1, cfg, 10)
    tree2 = make_tree(m2, cfg, 10)
    print('n_nodes: ', str(len(tree1.nodes)))

    fmm = FMM(tree1, m1, tree2, m2, cfg)
    fmmeval = FMMEvaluator(fmm)
    full = op(m1, m2, K = K)

    x = np.random.rand(full.shape[1])
    y1 = full.dot(x)

    x_tree = fmm.to_tree(x)
    import taskloaf as tsk
    async def call_fmm(tsk_w):
        return (await fmmeval.eval(tsk_w, x_tree, return_all_intermediates = True))
    fmm_res, m_check, multipoles, l_check, locals = tsk.run(call_fmm)
    y2 = fmm.to_orig(fmm_res)
    np.testing.assert_almost_equal(y1, y2)

def test_tri_fmm_m2p_single():
    np.random.seed(100)

    n = 10
    m1 = mesh_gen.make_rect(n, n, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    m2 = mesh_gen.make_rect(n, n, [[-10, 0, 1], [-10, 0, -1], [-8, 0, -1], [-8, 0, 1]])

    K = 'elasticRH3'
    cfg = make_config(K, [1.0, 0.25], 1.1, 2.5, 2, np.float32, treecode = True, force_order = 1)
    tree1 = make_tree(m1, cfg, 1000)
    tree2 = make_tree(m2, cfg, 1000)

    src_R = tree2.nodes[0].bounds.R
    center = tree2.nodes[0].bounds.center
    R_outer = cfg.outer_r
    R_inner = cfg.inner_r

    scaling = 1.0
    check_sphere = mesh_gen.make_sphere(center, scaling * R_outer * src_R, 2)
    equiv_sphere = mesh_gen.make_sphere(center, scaling * R_inner * src_R, 2)

    src_tri_idxs = tree2.orig_idxs
    src_tris = m2[1][src_tri_idxs]
    obs_tri_idxs = tree1.orig_idxs
    obs_tris = m1[1][obs_tri_idxs]

    p2c = op(check_sphere, (m2[0], src_tris), K)

    e2c = op(check_sphere, equiv_sphere, K, nq = 4)
    c2e = reg_lstsq_inverse(e2c, cfg.alpha)

    e2p = op((m1[0], obs_tris), equiv_sphere, K)

    p2p = op((m1[0], obs_tris), (m2[0], src_tris), K)
    fmm_mat = e2p.dot(c2e.dot(p2c))

    full = op(m1, m2, K = K)

    fmm = FMM(tree1, m1, tree2, m2, cfg)
    fmmeval = FMMEvaluator(fmm)

    x = np.random.rand(full.shape[1])
    y1 = full.dot(x)

    x_tree = fmm.to_tree(x)
    import taskloaf as tsk
    async def call_fmm(tsk_w):
        return (await fmmeval.eval(tsk_w, x_tree, return_all_intermediates = True))
    fmm_res, m_check, multipoles, l_check, locals = tsk.run(call_fmm)
    y2 = fmm.to_orig(fmm_res)

    m_check2 = p2c.dot(x_tree)
    np.testing.assert_almost_equal(m_check, m_check2)
    np.testing.assert_almost_equal(multipoles, c2e.dot(m_check))
    np.testing.assert_almost_equal(fmm_res, e2p.dot(multipoles), 4)
    np.testing.assert_almost_equal(y1, y2, 5)

def test_tri_fmm_full():
    np.random.seed(100)

    n = 40
    K = 'elasticRA3'
    m1 = mesh_gen.make_rect(n, n, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    m2 = mesh_gen.make_rect(n, n, [[-3, 0, 1], [-3, 0, -1], [-2, 0, -1], [-2, 0, 1]])
    x = np.random.rand(m2[1].shape[0] * 9)

    t = Timer()

    cfg = make_config(K, [1.0, 0.25], 1.1, 4.5, 2, np.float32)
    tree1 = make_tree(m1, cfg, 100)
    tree2 = make_tree(m2, cfg, 100)
    print('n_nodes: ', str(len(tree1.nodes)))

    fmm = FMM(tree1, m1, tree2, m2, cfg)
    report_interactions(fmm)
    fmmeval = FMMEvaluator(fmm)
    t.report('setup fmm')

    full = op(m1, m2, K = K)
    t.report('setup dense')

    t.report('gen x')
    y1 = full.dot(x)
    t.report('dense mv')

    x_tree = fmm.to_tree(x)
    import taskloaf as tsk
    async def call_fmm(tsk_w):
        return (await fmmeval.eval(tsk_w, x_tree))
    fmm_res = tsk.run(call_fmm)
    y2 = fmm.to_orig(fmm_res)
    t.report('fmm mv')

    np.testing.assert_almost_equal(y1, y2)

@golden_master(digits = 5)
def test_c2e(request):
    n = 10
    m1 = mesh_gen.make_rect(n, n, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    m2 = mesh_gen.make_rect(n, n, [[-3, 0, 1], [-3, 0, -1], [-2, 0, -1], [-2, 0, 1]])

    K = 'elasticRH3'
    t = Timer()
    cfg = make_config(K, [1.0, 0.25], 1.1, 2.5, 2, np.float64, treecode = True)
    tree1 = make_tree(m1, cfg, 100)
    tree2 = make_tree(m2, cfg, 100)
    print(len(tree1.nodes))
    fmm = FMM(tree1, m1, tree2, m2, cfg)

    u2e_ops = []
    for n in tree2.nodes:
        UT, eig, V = fmm.u2e_ops
        alpha = cfg.alpha
        R = n.bounds.R
        inv_eig = R * eig / ((R * eig) ** 2 + alpha ** 2)
        u2e_ops.append((V * inv_eig).dot(UT))
    return np.array(u2e_ops)

def benchmark():
    tct.logger.setLevel(logging.INFO)
    m = 1
    n = 50
    K = 'elasticRH3'
    m1 = mesh_gen.make_rect(n, n, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    m2 = mesh_gen.make_rect(n, n, [[-3, 0, 1], [-3, 0, -1], [-2, 0, -1], [-2, 0, 1]])
    x = np.random.rand(m2[1].shape[0] * 9)

    t = Timer()
    new_pts, new_tris = concat(m1, m2)
    n_obs_tris = m1[1].shape[0]
    sparse_op = RegularizedSparseIntegralOp(
        8,8,8,2,5,2.5,K,K,[1.0, 0.25],new_pts,new_tris,
        np.float32,TriToTriDirectFarfieldOp,
        obs_subset = np.arange(0,n_obs_tris),
        src_subset = np.arange(n_obs_tris,new_tris.shape[0])
    )
    t.report('assemble matrix free')
    for i in range(m):
        y1 = sparse_op.dot(x)
    t.report('matrix free mv x10')

    fmm = FMMFarfieldOp(mac = 4.5, pts_per_cell = 300)(
        2, K, [1.0, 0.25], new_pts, new_tris, np.float32,
        obs_subset = np.arange(0,n_obs_tris),
        src_subset = np.arange(n_obs_tris,new_tris.shape[0])
    )
    report_interactions(fmm.fmm_obj)
    t.report('setup fmm')

    y2 = fmm.dot(x)
    t.report('fmm mv x10')

    print(y1, y2)


if __name__ == "__main__":
    benchmark()
