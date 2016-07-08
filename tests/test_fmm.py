import sys
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

import tectosaur.mesh as mesh
from tectosaur.util.timer import Timer

from test_decorators import slow

import cppimport
fmm = cppimport.imp("tectosaur.fmm.fmm").fmm.fmm

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

# def plot_matrix(pts, src_pts, p2p, p2m, m2p, m2m):
#     dense = np.empty((pts.shape[0] + m2m.shape[0], src_pts.shape[0] + m2m.shape[1]))
#     dense[:pts.shape[0],:src_pts.shape[0]] = p2p.todense()
#     dense[pts.shape[0]:,:src_pts.shape[0]] = p2m.todense()
#     dense[:pts.shape[0],src_pts.shape[0]:] = m2p.todense()
#     dense[pts.shape[0]:,src_pts.shape[0]:] = m2m.todense()
#     import matplotlib.pyplot as plt
#     plt.spy(dense)
#     plt.show()

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

    n_surf = fmm_mat.translation_surface_order
    n_inputs = src_pts.shape[0] * tdim
    n_outputs = obs_pts.shape[0] * tdim
    n_multipoles = n_surf * len(src_kd.nodes) * tdim
    n_locals = n_surf * len(obs_kd.nodes) * tdim
    input_vals = np.ones(n_inputs)
    t.report("make input")

    t2 = Timer()
    est = fmm_mat.p2p.matvec(input_vals, n_outputs)
    t.report("p2p")

    m_check = fmm_mat.p2m.matvec(input_vals, n_multipoles)
    multipoles = fmm_mat.uc2e[0].matvec(m_check, n_multipoles)

    t.report("p2m")
    for m2m, uc2e in zip(fmm_mat.m2m[1:], fmm_mat.uc2e[1:]):
        m_check = m2m.matvec(multipoles, n_multipoles)
        multipoles += uc2e.matvec(m_check, n_multipoles)
    t.report("m2m")

    l_check = fmm_mat.p2l.matvec(input_vals, n_locals)
    t.report("p2l")
    l_check += fmm_mat.m2l.matvec(multipoles, n_locals)
    t.report("m2l")

    locals = np.zeros(n_locals)
    for l2l, dc2e in zip(fmm_mat.l2l, fmm_mat.dc2e):
        l_check += l2l.matvec(locals, n_locals)
        locals += dc2e.matvec(l_check, n_locals)
    t.report("l2l")

    est += fmm_mat.m2p.matvec(multipoles, n_outputs)
    t.report("m2p")

    est += fmm_mat.l2p.matvec(locals, n_outputs)
    t.report("l2p")
    t2.report("matvec")

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

def check_invr(obs_pts, src_pts, est, accuracy = 3):
    correct_matrix = 1.0 / (scipy.spatial.distance.cdist(obs_pts, src_pts))
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
        check_invr(obs_pts, src_pts, est[d::3] / 3.0)

def test_double_layer():
    obs_pts, obs_ns, src_pts, src_ns, est = run_full(
        6000, rand_pts, 3.0, 45, "laplace_double", []
    )
    correct_mat = fmm.direct_eval(
        "laplace_double", obs_pts, obs_ns, src_pts, src_ns, []
    ).reshape((obs_pts.shape[0], src_pts.shape[0]))
    correct = correct_mat.dot(np.ones(src_pts.shape[0]))
    check(est, correct, 3)

def test_elasticH():
    params = [1.0, 0.25]
    K = "elasticT"
    obs_pts, obs_ns, src_pts, src_ns, est = run_full(
        35000, ellipse_pts, 2.8, 52, K, params
    )
    # correct_mat = fmm.direct_eval(
    #     K, obs_pts, obs_ns, src_pts, src_ns, params
    # ).reshape((3 * obs_pts.shape[0], 3 * src_pts.shape[0]))
    # correct = correct_mat.dot(np.ones(3 * src_pts.shape[0]))
    # check(est, correct, 3)


if __name__ == '__main__':
    # test_build_big()
    test_elasticH()
