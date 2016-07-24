import sys
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

import tectosaur.mesh as mesh
from tectosaur.util.timer import Timer

from test_decorators import slow

import tectosaur.fmm as fmm

def rand_pts(n, source):
    return np.random.rand(n, 3)

def ellipse_pts(n, source):
    a = 4.0
    b = 1.0
    c = 1.0
    uv = np.random.rand(n, 2)
    uv[:, 0] = (uv[:, 0] * np.pi) - np.pi / 2
    uv[:, 1] = (uv[:, 1] * 2 * np.pi) - np.pi
    x = a * np.cos(uv[:, 0]) * np.cos(uv[:, 1])
    y = b * np.cos(uv[:, 0]) * np.sin(uv[:, 1])
    z = c * np.sin(uv[:, 0])
    return np.array([x, y, z]).T

def test_fmm_cpp():
    fmm.run_tests([])

def test_kdtree_bisects():
    pts = np.random.rand(100,3)
    kdtree = fmm.KDTree(pts, pts, 1)
    pts = np.array(kdtree.pts)
    for n in kdtree.nodes:
        if n.is_leaf:
            continue
        l = kdtree.nodes[n.children[0]]
        r = kdtree.nodes[n.children[1]]
        assert(l.start == n.start)
        assert(r.end == n.end)
        assert(l.end == r.start)

def test_kdtree_contains_pts():
    pts = np.random.rand(100,3)
    kdtree = fmm.KDTree(pts, pts, 1)
    pts = np.array(kdtree.pts)
    for n in kdtree.nodes:
        for d in range(3):
            dist = np.sqrt(
                np.sum((pts[n.start:n.end,:] - n.bounds.center) ** 2, axis = 1)
            )
            assert(np.all(dist < n.bounds.r * 1.0001))

def test_kdtree_height_depth():
    pts = np.random.rand(100,3)
    kdtree = fmm.KDTree(pts, pts, 1)
    for n in kdtree.nodes:
        if n.is_leaf:
            continue
        for c in range(2):
            assert(n.depth == kdtree.nodes[n.children[c]].depth - 1)
        assert(n.height ==
            max([kdtree.nodes[n.children[c]].height for c in range(2)]) + 1)

def test_kdtree_orig_idx():
    pts = np.random.rand(100,3)
    kdtree = fmm.KDTree(pts, pts, 1)
    for i, orig_i in enumerate(kdtree.orig_idxs):
        np.testing.assert_almost_equal(kdtree.pts[i], pts[orig_i, :], 10)

def test_kdtree_idx():
    pts = np.random.rand(100,3)
    kdtree = fmm.KDTree(pts, pts, 1)
    for i, n in enumerate(kdtree.nodes):
        assert(n.idx == i)

@slow
def test_build_big():
    pts = np.random.rand(1000000, 3)
    import time
    start = time.time()
    kdtree = fmm.KDTree(pts, pts, 1)
    print("KDTree took: " + str(time.time() - start))

def run_full(n, make_pts, mac, order, kernel, params):
    obs_pts = make_pts(n, False)
    obs_ns = make_pts(n, False)
    obs_ns /= np.linalg.norm(obs_ns, axis = 1)[:,np.newaxis]
    src_pts = make_pts(n + 1, True)
    src_ns = make_pts(n + 1, True)
    src_ns /= np.linalg.norm(src_ns, axis = 1)[:,np.newaxis]

    t = Timer()
    obs_kd = fmm.KDTree(obs_pts, obs_ns, order)
    src_kd = fmm.KDTree(src_pts, src_ns, order)
    fmm_mat = fmm.fmmmmmmm(
        obs_kd, src_kd, fmm.FMMConfig(1.1, mac, order, kernel, params)
    )
    t.report("build matrices")

    tdim = fmm_mat.tensor_dim
    nnz = dict(
        p2p = fmm_mat.p2p.get_nnz(),
        p2m = fmm_mat.p2m.get_nnz(),
        p2l = fmm_mat.p2l.get_nnz(),
        m2p = fmm_mat.m2p.get_nnz(),
        m2m = sum([m.get_nnz() for m in fmm_mat.m2m]),
        m2l = fmm_mat.m2l.get_nnz(),
        l2p = fmm_mat.l2p.get_nnz(),
        l2l = sum([m.get_nnz() for m in fmm_mat.l2l]),
        uc2e = sum([m.get_nnz() for m in fmm_mat.uc2e]),
        dc2e = sum([m.get_nnz() for m in fmm_mat.dc2e])
    )
    for k,v in sorted(nnz.items(), key = lambda k: k[0])[::-1]:
        print(k + " nnz fraction: " + str(v / ((tdim * n) ** 2)))
    total_nnz = sum(nnz.values())
    print("total nnz: " + str(total_nnz))
    print("compression ratio: " + str(total_nnz / ((tdim * n) ** 2)))

    n_outputs = obs_pts.shape[0] * tdim
    input_vals = np.ones(src_pts.shape[0] * tdim)
    t.report("make input")

    est = fmm.eval(obs_kd, src_kd, fmm_mat, input_vals, n_outputs)
    t.report("matvec")

    return (
        np.array(obs_kd.pts), np.array(obs_kd.normals),
        np.array(src_kd.pts), np.array(src_kd.normals), est
    )

def test_ones():
    obs_pts, _, src_pts, _, est = run_full(5000, rand_pts, 0.5, 1, "one",[])
    assert(np.all(np.abs(est - 5001) < 1e-3))

def check(est, correct, accuracy):
    rmse = np.sqrt(np.mean((est - correct) ** 2))
    rms_c = np.sqrt(np.mean(correct ** 2))
    print("L2ERR: " + str(rmse / rms_c))
    print("MEANERR: " + str(np.mean(np.abs(est - correct)) / rms_c))
    print("MAXERR: " + str(np.max(np.abs(est - correct)) / rms_c))
    print("MEANRELERR: " + str(np.mean(np.abs((est - correct) / correct))))
    print("MAXRELERR: " + str(np.max(np.abs((est - correct) / correct))))
    np.testing.assert_almost_equal(est / rms_c, correct / rms_c, accuracy)

def check_invr(obs_pts, _0, src_pts, _1, est, accuracy = 3):
    correct_matrix = 1.0 / (scipy.spatial.distance.cdist(obs_pts, src_pts))
    correct_matrix[np.isnan(correct_matrix)] = 0
    correct_matrix[np.isinf(correct_matrix)] = 0

    correct = correct_matrix.dot(np.ones(src_pts.shape[0]))
    check(est, correct, accuracy)

def test_invr():
    check_invr(*run_full(5000, rand_pts, 2.6, 30, "invr", []))

@slow
def test_high_accuracy():
    check_invr(*run_full(15000, rand_pts, 2.6, 200, "invr", []), accuracy = 8)

def test_irregular():
    check_invr(*run_full(10000, ellipse_pts, 2.6, 35, "invr", []))

def test_tensor():
    obs_pts, _, src_pts, _, est = run_full(5000, rand_pts, 2.6, 35, "tensor_invr", [])
    for d in range(3):
        check_invr(obs_pts, _, src_pts, _, est[d::3] / 3.0)

def test_double_layer():
    obs_pts, obs_ns, src_pts, src_ns, est = run_full(
        6000, rand_pts, 3.0, 45, "laplace_double", []
    )
    correct_mat = fmm.direct_eval(
        "laplace_double", obs_pts, obs_ns, src_pts, src_ns, []
    ).reshape((obs_pts.shape[0], src_pts.shape[0]))
    correct = correct_mat.dot(np.ones(src_pts.shape[0]))
    check(est, correct, 3)

@slow
def test_elasticH():
    params = [1.0, 0.25]
    K = "elasticT"
    obs_pts, obs_ns, src_pts, src_ns, est = run_full(
        10000, ellipse_pts, 2.8, 52, K, params
    )
    # correct_mat = fmm.direct_eval(
    #     K, obs_pts, obs_ns, src_pts, src_ns, params
    # ).reshape((3 * obs_pts.shape[0], 3 * src_pts.shape[0]))
    # correct = correct_mat.dot(np.ones(3 * src_pts.shape[0]))
    # check(est, correct, 3)

def test_self_fmm():
    order = 60
    mac = 3.0
    n = 3000
    k_name = "elasticH"
    params = [1.0, 0.25]
    pts = np.random.rand(n, 3)
    ns = np.random.rand(n, 3)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
    kd = fmm.KDTree(pts, ns, order)
    fmm_mat = fmm.fmmmmmmm(
        kd, kd, fmm.FMMConfig(1.1, mac, order, k_name, params)
    )
    est = fmm.eval(kd, kd, fmm_mat, np.ones(n * 3), n * 3)
    correct_mat = fmm.direct_eval(
        k_name, np.array(kd.pts), np.array(kd.normals),
        np.array(kd.pts), np.array(kd.normals), params
    ).reshape((n * 3, n * 3))
    correct_mat[np.isnan(correct_mat)] = 0
    correct_mat[np.isinf(correct_mat)] = 0
    correct = correct_mat.dot(np.ones(n * 3))
    check(est, correct, 2)


if __name__ == '__main__':
    # test_build_big()
    test_elasticH()
