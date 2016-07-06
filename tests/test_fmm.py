import sys
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

import tectosaur.mesh as mesh
from tectosaur.util.timer import Timer

from test_decorators import slow

import cppimport
cppimport.set_quiet(False)
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

def surrounding_surface_sphere(order):
    pts = []
    a = 4 * np.pi / order;
    d = np.sqrt(a);
    M_theta = int(np.round(np.pi / d))
    d_theta = np.pi / M_theta;
    d_phi = a / d_theta;
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta;
        M_phi = int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi;
            x = np.sin(theta) * np.cos(phi);
            y = np.sin(theta) * np.sin(phi);
            z = np.cos(theta);
            pts.append((x, y, z))
    return np.array(pts)

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
    assert(kdtree.max_depth == kdtree.max_height)
    for n in kdtree.nodes:
        if n.is_leaf:
            continue
        for c in range(2):
            assert(n.depth == kdtree.nodes[n.children[c]].depth - 1)
        assert(n.height ==
            max([kdtree.nodes[n.children[c]].height for c in range(2)]) + 1)

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

# def plot_matrix(pts, pts2, p2p, p2m, m2p, m2m):
#     dense = np.empty((pts.shape[0] + m2m.shape[0], pts2.shape[0] + m2m.shape[1]))
#     dense[:pts.shape[0],:pts2.shape[0]] = p2p.todense()
#     dense[pts.shape[0]:,:pts2.shape[0]] = p2m.todense()
#     dense[:pts.shape[0],pts2.shape[0]:] = m2p.todense()
#     dense[pts.shape[0]:,pts2.shape[0]:] = m2m.todense()
#     import matplotlib.pyplot as plt
#     plt.spy(dense)
#     plt.show()

def run_full(n, make_pts, mac, order, kernel):
    pts = make_pts(n, False)
    pts2 = make_pts(n + 1, True)

    t = Timer()
    kd = fmm.KDTree(pts, pts, order)
    kd2 = fmm.KDTree(pts2, pts2, order)
    surf = surrounding_surface_sphere(order)
    fmm_mat = fmm.fmmmmmmm(kd, kd2, fmm.FMMConfig(1.1, mac, surf, kernel))
    t.report("build matrices")

    n_src = pts.shape[0]
    n_obs = pts2.shape[0]

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
        print(k + " nnz fraction: " + str(v / (n ** 2)))
    total_nnz = sum(nnz.values())
    print("total nnz: " + str(total_nnz))
    print("compression ratio: " + str(total_nnz / (n ** 2)))

    input_vals = np.ones(pts2.shape[0])
    n_multipoles = surf.shape[0] * len(kd2.nodes)
    n_locals = surf.shape[0] * len(kd.nodes)
    t.report("make input")

    t2 = Timer()
    est = fmm_mat.p2p.matvec(input_vals, pts.shape[0])
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

    est += fmm_mat.m2p.matvec(multipoles, pts.shape[0])
    t.report("m2p")

    est += fmm_mat.l2p.matvec(locals, pts.shape[0])
    t.report("l2p")
    t2.report("matvec")

    return np.array(kd.pts), np.array(kd2.pts), est

def test_ones():
    pts, pts2, est = run_full(5000, rand_pts, 0.5, 1, "one")
    assert(np.all(np.abs(est - 5001) < 1e-3))

def check_invr(pts, pts2, est, accuracy = 3):
    correct_matrix = 1.0 / (scipy.spatial.distance.cdist(pts, pts2))
    correct = correct_matrix.dot(np.ones(pts2.shape[0]))
    error = np.sqrt(np.mean((est - correct) ** 2))
    print("L2ERR: " + str(error / np.mean(correct)))
    print("MEANERR: " + str(np.mean(np.abs(est - correct)) / np.mean(correct)))
    print("MAXERR: " + str(np.max(np.abs(est - correct) / correct)))
    np.testing.assert_almost_equal(
        est / np.mean(correct), correct / np.mean(correct), accuracy
    )

def test_invr():
    check_invr(*run_full(5000, rand_pts, 2.6, 30, "invr"), accuracy = 3)

@slow
def test_high_accuracy():
    check_invr(*run_full(15000, rand_pts, 2.6, 200, "invr"), accuracy = 8)

def test_irregular():
    check_invr(*run_full(10000, ellipse_pts, 2.6, 35, "invr"))



if __name__ == '__main__':
    # test_build_big()
    test_irregular()
