import numpy as np
from tectosaur.ops.sparse_farfield_op import (
    TriToTriDirectFarfieldOp, PtToPtFMMFarfieldOp, PtToPtDirectFarfieldOp)
from tectosaur.ops.dense_integral_op import farfield_tris
import tectosaur.mesh.mesh_gen as mesh_gen
from tectosaur.kernels import kernels
from tectosaur.mesh.modify import concat
import tectosaur.util.gpu as gpu
from tectosaur.util.quadrature import gauss4d_tri
from test_farfield import make_meshes
from tectosaur.fmm.fmm import make_tree, make_config, FMM, FMMEvaluator, report_interactions
from tectosaur.fmm.c2e import reg_lstsq_inverse
from tectosaur.constraint_builders import continuity_constraints
from tectosaur.constraints import build_constraint_matrix

# TODO: dim
def test_tri_ones():
    m, surf1_idxs, surf2_idxs = make_meshes(n_m = 2, sep = 5, w = 1)
    ops = [
        C(
            3, 'tensor_one3', [], m[0], m[1], np.float64,
            surf1_idxs, surf2_idxs
        ) for C in [
            TriToTriDirectFarfieldOp,
            PtToPtDirectFarfieldOp,
            PtToPtFMMFarfieldOp(250, 3.0, 250)
        ]
    ]

    x = np.random.rand(ops[0].shape[1])
    x = np.ones(ops[0].shape[1])
    ys = [op.dot(x) for op in ops]
    for y in ys:
        print(y)

from tectosaur.util.cpp import imp
traversal_ext = imp("tectosaur.fmm.traversal_wrapper")

def check_children(tree, n):
    if n.is_leaf:
        return
    n_total = 0
    n_correct = n.end - n.start
    for c in n.children:
        if c == 0:
            continue
        n_c = tree.nodes[c].end - tree.nodes[c].start
        n_total += n_c
    assert(n_total == n_correct)
    for c in n.children:
        if c == 0:
            continue
        check_children(tree, tree.nodes[c])

def op(obs_m, src_m, K = 'elasticRH3', nq = 2):
    new_pts, new_tris = concat(obs_m, src_m)
    n_obs_tris = obs_m[1].shape[0]
    obs_tris = new_tris[:n_obs_tris]
    src_tris = new_tris[n_obs_tris:]
    mat = farfield_tris(
        K, [1.0, 0.25], new_pts, obs_tris, src_tris, nq, np.float32
    )
    nrows = mat.shape[0] * 9
    ncols = mat.shape[3] * 9
    return mat.reshape((nrows, ncols))

def get_gpu_module(surf, q, K, float_type, n_workers_per_block, n_c2e_block_rows):
    args = dict(
        n_workers_per_block = n_workers_per_block,
        n_c2e_block_rows = n_c2e_block_rows,
        gpu_float_type = gpu.np_to_c_type(float_type),
        surf_pts = surf[0],
        surf_tris = surf[1],
        quad_pts = q[0],
        quad_wts = q[1],
        K = K
    )
    gpu_module = gpu.load_gpu(
        'fmm/tri_gpu_kernels.cl',
        tmpl_args = args
    )
    return gpu_module

float_type = np.float32
sphere = mesh_gen.make_sphere((0.0, 0.0, 0.0), 1.0, 2)
def get_standard_fmm_gpu_module(K_name):
    q = gauss4d_tri(3, 3)
    K = kernels[K_name]
    return get_gpu_module(sphere, q, K, float_type, 64, 16)

def test_cuda_tri_fmm():
    get_standard_fmm_gpu_module()

def gpu_p2s(center, node_R, surf_R, src_m, src_in):
    module = get_standard_fmm_gpu_module('elasticRH3')
    gpu_src_n_starts = gpu.to_gpu(np.array([0]), np.int32)
    gpu_src_n_ends = gpu.to_gpu(np.array([src_m[1].shape[0]]), np.int32)
    gpu_src_pts = gpu.to_gpu(src_m[0].flatten(), float_type)
    gpu_src_tris = gpu.to_gpu(src_m[1].flatten(), np.int32)
    gpu_centers = gpu.to_gpu(np.array([center]).flatten(), float_type)
    gpu_Rs = gpu.to_gpu(np.array([node_R]).flatten(), float_type)

    gpu_obs_n_idxs = gpu.to_gpu(np.array([0]), np.int32)
    gpu_obs_src_starts = gpu.to_gpu(np.array([0, 1]), np.int32)
    gpu_src_n_idxs = gpu.to_gpu(np.array([0]), np.int32)

    n_blocks = 1
    n_workers_per_block = 64
    n_surf_tris = sphere[1].shape[0]
    gpu_out = gpu.zeros_gpu(n_surf_tris * 9, float_type)
    gpu_in = gpu.to_gpu(src_in, float_type)
    gpu_params = gpu.to_gpu(np.array([1.0, 0.25]), float_type)

    module.p2s_elasticRH3(
        gpu_out, gpu_in, np.int32(n_blocks), gpu_params,
        gpu_obs_n_idxs, gpu_obs_src_starts, gpu_src_n_idxs,
        gpu_centers, gpu_Rs, float_type(surf_R),
        gpu_src_n_starts, gpu_src_n_ends,
        gpu_src_pts, gpu_src_tris,
        grid = (n_blocks, 1, 1),
        block = (n_workers_per_block, 1, 1)
    )
    return gpu_out.get()


def test_tri_fmm_full():
    np.random.seed(100)

    n = 10
    m1 = mesh_gen.make_rect(n, n, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    m2 = mesh_gen.make_rect(n, n, [[-3, 0, 1], [-3, 0, -1], [-2, 0, -1], [-2, 0, 1]])


    K = 'elasticRH3'

    cfg = make_config(K, [1.0, 0.25], 1.1, 2.5, 2, np.float32, treecode = True)
    tree1 = make_tree(m1, cfg, 10)
    tree2 = make_tree(m2, cfg, 10)
    fmm = FMM(tree1, m1, tree1, m2, cfg)
    report_interactions(fmm)
    fmmeval = FMMEvaluator(fmm)
    full = op(m1, m2, K = K)

    cs = continuity_constraints(m1[1], np.array([]))
    cm, c_rhs = build_constraint_matrix(cs, full.shape[1])

    x = cm.dot(np.random.rand(cm.shape[1]))
    y1 = cm.T.dot(full.dot(x))

    x_tree = fmm.to_tree(x)
    import taskloaf as tsk
    async def call_fmm(tsk_w):
        return (await fmmeval.eval(tsk_w, x_tree, return_all_intermediates = True))
    fmm_res, m_check, multipoles, l_check, locals = tsk.run(call_fmm)
    y2 = cm.T.dot(fmm.to_orig(fmm_res))

    print(y1[0], y2[0])

    # import matplotlib.pyplot as plt
    # plt.plot(y1)
    # plt.plot(y2)
    # plt.show()


    # Run with order = 10 and p2p off and only first node

    # obs_n = fmm.obs_tree.nodes[fmm.interactions.m2p.obs_n_idxs[0]]
    # src_n = fmm.src_tree.nodes[fmm.interactions.m2p.src_n_idxs[0]]
    # src_center = src_n.bounds.center
    # src_R = src_n.bounds.R

    # src_tri_idxs = tree2.orig_idxs[src_n.start:src_n.end]
    # src_tris = m2[1][src_tri_idxs]
    # obs_tri_idxs = tree1.orig_idxs[obs_n.start:obs_n.end]
    # obs_tris = m1[1][obs_tri_idxs]

    # check_sphere = mesh_gen.make_sphere(src_center, cfg.outer_r * src_R, 2)
    # equiv_sphere = mesh_gen.make_sphere(src_center, cfg.inner_r * src_R, 2)
    # e2p = op((m1[0], obs_tris), equiv_sphere)

    # multipole_start = src_n.idx * cfg.surf[1].shape[0] * 9
    # multipole_end = (src_n.idx + 1) * cfg.surf[1].shape[0] * 9

    # y3_chunk = e2p.dot(multipoles[multipole_start:multipole_end])
    # np.testing.assert_almost_equal(y3_chunk, fmm_res[:y3_chunk.shape[0]])

    for i in range(fmm.interactions.p2m.obs_n_idxs.shape[0]):
        src_n = fmm.src_tree.nodes[fmm.interactions.p2m.obs_n_idxs[i]]
        if src_n.end - src_n.start > 0:
            break

    src_center = src_n.bounds.center
    src_R = src_n.bounds.R

    src_tri_idxs = tree2.orig_idxs[src_n.start:src_n.end]
    src_tris = m2[1][src_tri_idxs]

    check_sphere = mesh_gen.make_sphere(src_center, cfg.outer_r * src_R, 2)
    equiv_sphere = mesh_gen.make_sphere(src_center, cfg.inner_r * src_R, 2)
    p2c = op(check_sphere, (m2[0], src_tris))
    m_check3 = p2c.dot(x_tree[(src_n.start * 9):(src_n.end * 9)])

    start_dof = src_n.idx * cfg.surf[1].shape[0] * 9
    end_dof = (src_n.idx + 1) * cfg.surf[1].shape[0] * 9
    np.testing.assert_almost_equal(m_check3, m_check[start_dof:end_dof])

    e2c = op(check_sphere, equiv_sphere)
    c2e = reg_lstsq_inverse(e2c, 1e-5 / src_R)

    multipoles3 = c2e.dot(m_check3)
    calc = multipoles[start_dof:end_dof]
    calc2 = fmm.gpu_data['u2e_ops'].get().dot(m_check[start_dof:end_dof])
    import ipdb
    ipdb.set_trace()

def test_c2e_scaling():
    inR = 1.1
    outR = 3.0
    src_R = 2.0
    order = 2
    check_sphere1 = mesh_gen.make_sphere((0.0,0.0,0.0), outR * 1.0, order)
    equiv_sphere1 = mesh_gen.make_sphere((0.0,0.0,0.0), inR * 1.0, order)
    check_sphereR = mesh_gen.make_sphere((0.0,0.0,0.0), outR * src_R, order)
    equiv_sphereR = mesh_gen.make_sphere((0.0,0.0,0.0), inR * src_R, order)

    for nq in range(2,6):
        e2c1 = op(check_sphere1, equiv_sphere1, nq = nq)
        e2cR = op(check_sphereR, equiv_sphereR, nq = nq)

        cs = continuity_constraints(check_sphereR[1], np.array([]))
        cm, c_rhs = build_constraint_matrix(cs, e2c1.shape[1])

        e2c1_constrained = cm.T.dot((cm.T.dot(e2c1)).T).T
        e2cR_constrained = cm.T.dot((cm.T.dot(e2cR)).T).T

        alpha = 1e-5
        c2e1 = reg_lstsq_inverse(e2c1_constrained, alpha)
        c2eR = reg_lstsq_inverse(e2cR_constrained, alpha * src_R)

        x = np.random.rand(c2e1.shape[1])
        y1 = c2e1.dot(x)
        y2 = c2eR.dot(x)

        # print(alpha)
        print(c2e1[0,0], c2eR[0,0])
        # print(y1 / y2)
    import ipdb
    ipdb.set_trace()



def test_tri_kdtree():
    np.random.seed(100)

    n = 10
    m = mesh_gen.make_rect(n, n, [[-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
    pts, tris = m

    cfg = make_config('elasticRH3', [1.0, 0.25], 1.0, 3.0, 2, np.float32)
    tree = make_tree(m, cfg, 10)
    fmm = FMM(tree, m, tree, m, cfg)
    fmmeval = FMMEvaluator(fmm)

    check_children(tree, tree.nodes[0])


    R_outer = 3.0
    R_inner = 1.0

    interactions = traversal_ext.three.kdtree.fmmmm_interactions(
        tree, tree, R_inner, R_outer, 3, True
    )

    found = False
    for i in range(len(interactions.m2p.obs_n_idxs)):
        obs_idx = interactions.m2p.obs_n_idxs[i]
        start = interactions.m2p.obs_src_starts[i]
        end = interactions.m2p.obs_src_starts[i + 1]
        for j in range(start, end):
            src_idx = interactions.m2p.src_n_idxs[j]

            obs_n = tree.nodes[obs_idx]
            src_n = tree.nodes[src_idx]
            n_obs = obs_n.end - obs_n.start
            n_src = src_n.end - src_n.start
            if n_obs * n_src > 0:
                found = True
                break
        if found:
            break

    center = src_n.bounds.center
    R = src_n.bounds.R

    #TODO: use better sphere
    check_sphere = mesh_gen.make_sphere(center, R_outer * R, 2)
    equiv_sphere = mesh_gen.make_sphere(center, R_inner * R, 2)

    src_tri_idxs = tree.orig_idxs[src_n.start:src_n.end]
    src_tris = tris[src_tri_idxs]
    obs_tri_idxs = tree.orig_idxs[obs_n.start:obs_n.end]
    obs_tris = tris[obs_tri_idxs]

    p2c = op(check_sphere, (pts, src_tris))
    e2c = op(check_sphere, equiv_sphere)
    c2e = reg_lstsq_inverse(e2c, 5e-7)
    e2p = op((pts, obs_tris), equiv_sphere)

    p2p = op((pts, obs_tris), (pts, src_tris))

    fmm_mat = e2p.dot(c2e.dot(p2c))
    correct = p2p.reshape(fmm_mat.shape)
    for i in range(10):
        x = np.random.rand(fmm_mat.shape[1])

        # gpu_p2c = gpu_p2s(center, R, R_outer, (pts, src_tris), x)
        # p2c2 = p2c.dot(x)
        # import ipdb
        # ipdb.set_trace()

        y1 = fmm_mat.dot(x)
        y2 = correct.dot(x)

        # print(np.vstack((y1 , y2)).T)
        print(np.mean(np.abs(y2 - y1)) / np.mean(np.abs(y1)))
