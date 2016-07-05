import sys
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

import tectosaur.mesh as mesh
from tectosaur.timer import Timer

from test_decorators import slow

import cppimport
fmm = cppimport.imp("tectosaur.fmm").fmm

def make_line_pts(n):
    return np.array([np.linspace(0.0, 1.0, n+2)[1:-1], np.zeros(n), np.zeros(n)]).T

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

def get_nnz(fmm_mat):
    total_nnz = 0
    mats = []
    for mat in [fmm_mat.p2p, fmm_mat.p2m, fmm_mat.m2p, fmm_mat.m2m]:
        total_nnz += mat.get_nnz()
        print(mat.nnz)
    print("total nnz: " + str(total_nnz))

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

def test_p2m():
    pts1 = np.random.rand(100,3)
    pts2 = np.random.rand(100,3)
    pts2[:,0] += 2
    kd1 = fmm.KDTree(pts1, pts1, 1)
    kd2 = fmm.KDTree(pts2, pts2, 1)
    surf = surrounding_surface_sphere(1)
    fmm_mat = fmm.fmmmmmmm(kd1, kd2, fmm.FMMConfig(1.1, 1.9, surf, "one"))
    m_lowest = fmm_mat.p2m.matvec(np.ones(100), surf.shape[0] * len(kd2.nodes))
    for n in kd2.nodes:
        if not n.is_leaf:
            continue
        start_mdof = n.idx * surf.shape[0]
        end_mdof = start_mdof + surf.shape[0]
        assert(int(np.round(np.sum(m_lowest[start_mdof:end_mdof]))) == (n.end - n.start))

def test_m2m_m2p():
    while True:
        pts1 = np.random.rand(100,3)
        pts2 = np.random.rand(105,3)
        pts2[:,0] += 2.0
        kd1 = fmm.KDTree(pts1, pts1, 2)
        kd2 = fmm.KDTree(pts2, pts2, 2)
        surf = surrounding_surface_sphere(10)
        fmm_mat = fmm.fmmmmmmm(kd1, kd2, fmm.FMMConfig(1.1, 2.9, surf, "one"))

        result = fmm_mat.p2p.matvec(np.ones(pts2.shape[0]), pts1.shape[0])
        if np.any(np.abs(result) > 1e-5):
            continue

        n_multipoles = surf.shape[0] * len(kd2.nodes)
        multipoles = fmm_mat.p2m.matvec(
            np.ones(pts2.shape[0]), n_multipoles
        )
        for mat in fmm_mat.m2m[::-1]:
            multipoles += mat.matvec(multipoles, n_multipoles)

        n_surf = surf.shape[0]
        for n in kd2.nodes:
            start_mdof = n.idx * n_surf
            end_mdof = start_mdof + surf.shape[0]
            m_strength = int(np.round(np.sum(multipoles[start_mdof:end_mdof])))
            n_pts = n.end - n.start
            assert(m_strength == n_pts)

        result = fmm_mat.m2p.matvec(multipoles, pts1.shape[0])
        assert(np.all(np.abs(result - pts2.shape[0]) < 1e-3))
        break


def plot_matrix(pts, pts2, p2p, p2m, m2p, m2m):
    dense = np.empty((pts.shape[0] + m2m.shape[0], pts2.shape[0] + m2m.shape[1]))
    dense[:pts.shape[0],:pts2.shape[0]] = p2p.todense()
    dense[pts.shape[0]:,:pts2.shape[0]] = p2m.todense()
    dense[:pts.shape[0],pts2.shape[0]:] = m2p.todense()
    dense[pts.shape[0]:,pts2.shape[0]:] = m2m.todense()
    import matplotlib.pyplot as plt
    plt.spy(dense)
    plt.show()

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
        l2l = sum([m.get_nnz() for m in fmm_mat.l2l])
    )
    for k,v in sorted(nnz.items(), key = lambda k: k[0])[::-1]:
        print(k + " nnz fraction: " + str(v / (n ** 2)))
    total_nnz = sum(nnz.values())
    print("total nnz: " + str(total_nnz))
    print("compression ratio: " + str(total_nnz / (n ** 2)))

    # plot_matrix( pts, pts2, p2p p2m, m2p m2m)

    input_vals = np.ones(pts2.shape[0])
    n_multipoles = surf.shape[0] * len(kd2.nodes)
    n_locals = surf.shape[0] * len(kd.nodes)
    t.report("make input")

    t2 = Timer()
    est = fmm_mat.p2p.matvec(input_vals, pts.shape[0])
    t.report("p2p")

    multipoles = fmm_mat.p2m.matvec(input_vals, n_multipoles)
    t.report("p2m")
    for mat in fmm_mat.m2m[::-1]:
        multipoles += mat.matvec(multipoles, n_multipoles)
    t.report("m2m")

    locals = fmm_mat.p2l.matvec(input_vals, n_locals)
    t.report("p2l")
    locals += fmm_mat.m2l.matvec(multipoles, n_locals)
    t.report("m2l")

    for mat in fmm_mat.l2l:
        locals += mat.matvec(locals, n_locals)
    t.report("l2l")


    est += fmm_mat.m2p.matvec(multipoles, pts.shape[0])
    t.report("m2p")

    est += fmm_mat.l2p.matvec(locals, pts.shape[0])
    t.report("l2p")
    t2.report("matvec")

    return np.array(kd.pts), np.array(kd2.pts), est

def rand_pts(n, source):
    return np.random.rand(n, 3)

def test_ones():
    pts, pts2, est = run_full(5000, rand_pts, 0.5, 1, "one")
    assert(np.all(np.abs(est - 5001) < 1e-3))

def test_inv_r():
    pts, pts2, est = run_full(5000, rand_pts, 0.5, 30, "invr")
    correct = (1.0 / (scipy.spatial.distance.cdist(pts, pts2))).dot(np.ones(pts2.shape[0]))
    error = np.sqrt(np.mean((est - correct) ** 2))
    print("relerr: " + str(error / np.mean(correct)))
    np.testing.assert_almost_equal(
        est / np.mean(correct), correct / np.mean(correct), 3
    )

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

def test_irregular():
    pts, pts2, est = run_full(27000, ellipse_pts, 2.6, 35, "invr")
    correct = (1.0 / (scipy.spatial.distance.cdist(pts, pts2))).dot(np.ones(pts2.shape[0]))
    error = np.sqrt(np.mean((est - correct) ** 2))
    print("L2ERR: " + str(error / np.mean(correct)))
    print("MEANERR: " + str(np.mean(np.abs(est - correct)) / np.mean(correct)))
    print("MAXERR: " + str(np.max(np.abs(est - correct) / correct)))
    np.testing.assert_almost_equal(
        est / np.mean(correct), correct / np.mean(correct), 3
    )

@slow
def test_build_big():
    pts = np.random.rand(10000000, 3)
    import time
    start = time.time()
    kdtree = fmm.KDTree(pts, pts, 1)
    print("KDTree took: " + str(time.time() - start))

if __name__ == '__main__':
    # test_build_big()
    test_irregular()
