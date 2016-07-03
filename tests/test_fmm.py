import sys
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

import tectosaur.mesh as mesh
from tectosaur.timer import Timer

from test_decorators import slow

import taskloaf
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

def to_scipy(m, shape = None):
    if len(m.get_vals()) == 0:
        return scipy.sparse.coo_matrix((0,0))
    return scipy.sparse.coo_matrix((m.get_vals(), (m.get_rows(), m.get_cols())), shape)

def get_nnz(fmm_mat):
    total_nnz = 0
    mats = []
    for mat in [fmm_mat.p2p, fmm_mat.p2m, fmm_mat.m2p, fmm_mat.m2m]:
        mat = to_scipy(mat)
        total_nnz += mat.nnz
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
    fmm_mat = fmm.fmmmmmmm(kd1, kd2, fmm.FMMConfig(0.3, 1.1, 2.9, surf, "one"))
    p2m = to_scipy(fmm_mat.p2m)
    m_lowest = p2m.dot(np.ones(p2m.shape[1]))
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
        fmm_mat = fmm.fmmmmmmm(kd1, kd2, fmm.FMMConfig(0.4, 1.1, 2.9, surf, "one"))

        if len(fmm_mat.p2p.get_vals()) > 0:
            continue

        m2m = to_scipy(fmm_mat.m2m)
        m_lowest = to_scipy(fmm_mat.p2m, (m2m.shape[1], pts2.shape[0]))\
            .dot(np.ones(pts2.shape[0]))
        m_all = scipy.sparse.linalg.spsolve(m2m, -m_lowest)
        n_surf = surf.shape[0]
        for n in kd2.nodes:
            start_mdof = n.idx * n_surf
            end_mdof = start_mdof + surf.shape[0]
            m_strength = int(np.round(np.sum(m_all[start_mdof:end_mdof])))
            n_pts = n.end - n.start
            assert(m_strength == n_pts)

        m2p = to_scipy(fmm_mat.m2p, (pts1.shape[0], m_all.shape[0]))
        result = m2p.dot(m_all)
        assert(np.all(np.abs(result - pts2.shape[0]) < 1e-3))
        break


def plot_matrix():
    pass
    # dense = np.empty((pts.shape[0] + m2m.shape[0], pts2.shape[0] + m2m.shape[1]))
    # dense[:pts.shape[0],:pts2.shape[0]] = p2p.todense()
    # dense[pts.shape[0]:,:pts.shape[0]] = p2m.todense()
    # dense[:pts.shape[0],pts2.shape[0]:] = m2p.todense()
    # dense[pts.shape[0]:,pts2.shape[0]:] = m2m.todense()
    # import matplotlib.pyplot as plt
    # plt.spy(dense)
    # plt.show()

    # import ipdb; ipdb.set_trace()

def run_full(n, make_pts, mac, order, kernel):
    pts = make_pts(n, False)
    pts2 = make_pts(n + 1, True)

    t = Timer()
    kd = fmm.KDTree(pts, pts, order)
    kd2 = fmm.KDTree(pts2, pts2, order)
    surf = surrounding_surface_sphere(order)
    fmm_mat = fmm.fmmmmmmm(kd, kd2, fmm.FMMConfig(mac, 1.1, mac, surf, kernel))
    t.report("build matrices")

    n_src = pts.shape[0]
    n_obs = pts2.shape[0]

    # p2p = to_scipy(fmm_mat.p2p, (pts.shape[0], pts2.shape[0]))
    m2m = to_scipy(fmm_mat.m2m)
    p2m = to_scipy(fmm_mat.p2m, (m2m.shape[1], pts2.shape[0]))
    # m2p = to_scipy(fmm_mat.m2p, (pts.shape[0], m2m.shape[0]))
    print("p2p: " + str(fmm_mat.p2p.get_nnz() / (n ** 2)))
    print("p2m: " + str(p2m.nnz / (n ** 2)))
    print("m2m: " + str(m2m.nnz / (n ** 2)))
    print("m2p: " + str(fmm_mat.m2p.get_nnz() / (n ** 2)))
    nnz = (fmm_mat.p2p.get_nnz() + p2m.nnz + m2m.nnz + fmm_mat.m2p.get_nnz())
    print("compression ratio: " + str(nnz / (n ** 2)))
    t.report("copy to scipy")

    input_vals = np.ones(pts2.shape[0])
    t.report("make input")

    t2 = Timer()
    p2p_comp = fmm_mat.p2p.matvec(input_vals, pts.shape[0])
    est = p2p_comp
    t.report("p2p")
    if p2m.shape[1] > 0:
        m_lowest = p2m.dot(input_vals)
        t.report("p2m")
        m_all = scipy.sparse.linalg.spsolve(m2m, -m_lowest)
        t.report("m2m")
        m2p_comp = fmm_mat.m2p.matvec(m_all, pts.shape[0])
        t.report("m2p")
        est += m2p_comp
        t.report("sum")
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
    pts, pts2, est = run_full(5000, ellipse_pts, 0.9, 170, "invr")
    correct = (1.0 / (scipy.spatial.distance.cdist(pts, pts2))).dot(np.ones(pts2.shape[0]))
    error = np.sqrt(np.mean((est - correct) ** 2))
    print("L2ERR: " + str(error / np.mean(correct)))
    print("MEANERR: " + str(np.mean(np.abs(est - correct)) / np.mean(correct)))
    print("MAXERR: " + str(np.max(np.abs(est - correct) / correct)))
    np.testing.assert_almost_equal(
        est / np.mean(correct), correct / np.mean(correct), 3
    )

def invr_direct_mat(obs_pts, src_pts):
    return (1.0 / (scipy.spatial.distance.cdist(obs_pts, src_pts)))

def test_invr_multipole_accuracy():
    import matplotlib.pyplot as plt



    src_pts = np.random.rand(1000, 3) * 2.0 - 1.0
    src_r = np.sqrt(np.sum(src_pts ** 2, axis = 1))
    # plt.plot(src_r, '.')
    # plt.show()
    obs_pts = np.random.rand(10000, 3) * 20.0 - 10.0

    for pC in np.arange(5, 100, 5):
    # for C in np.linspace(2.0, 6.0, 20):
        # pC = 200
        # pE = 225
        pE = pC
        C = 2.0
        E = 1.1
        print(C, pC)
        check_surf = C * surrounding_surface_sphere(pC)
        equiv_surf = E * surrounding_surface_sphere(pE)

        src_to_check = invr_direct_mat(check_surf, src_pts)
        equiv_to_check = invr_direct_mat(check_surf, equiv_surf)
        # check_to_equiv = np.linalg.pinv(equiv_to_check, rcond = 1e-15)
        # SHOULD USE THE ORIGINAL FACTORS FROM THE SVD RATHER THAN COMPUTING
        # A PSEUDOINVERSE SINCE THIS MAINTAINS NUMERICAL PRECISION.
        c2e_svd = list(np.linalg.svd(equiv_to_check))
        c2e_svd[1] = c2e_svd[1] ** -1
        # src_to_equiv = check_to_equiv.dot(src_to_check)
        equiv_to_obs = invr_direct_mat(obs_pts, equiv_surf)
        # src_to_obs = equiv_to_obs.dot(src_to_equiv)
        obs_vals = equiv_to_obs.dot(
            c2e_svd[2].T.dot(
                np.diag(c2e_svd[1]).dot(
                    c2e_svd[0].T.dot(src_to_check.dot(np.ones(src_pts.shape[0]))))))
        # obs_vals = src_to_obs.dot(np.ones(src_pts.shape[0]))

        correct = invr_direct_mat(obs_pts, src_pts).dot(np.ones(src_pts.shape[0]))

        obs_r = np.sqrt(np.sum(obs_pts ** 2, axis = 1))
        obs_filtered = obs_vals[obs_r > C]
        cor_filtered = correct[obs_r > C]
        print("MEAN: " + str(np.mean(np.abs((obs_filtered - cor_filtered) / cor_filtered))))
        print("MAX: " + str(np.max(np.abs((obs_filtered - cor_filtered) / cor_filtered))))
        error = np.log10(np.abs((obs_vals - correct)))

    # plt.plot([C,C],[-50, 50])
    # plt.plot([E,E],[-50, 50])
    # plt.plot(obs_r, error, '.b')
    # plt.plot(obs_r, correct, '.r')
    # plt.ylim([np.min(error), np.max(error)])
    # plt.show()
    return

    def make_multipole_tester_pts(n, source):
        if source:
            return np.random.rand(n, 3) * 2 - 1.0
        else:
            return np.random.rand(n * 10, 3) * 30.0 - 15.0
    for p in range(2, 250, 20):
        pts, pts2, est = run_full(500, make_multipole_tester_pts, 100000000, p, "invr")
        correct = invr_direct_mat(pts, pts2).dot(np.ones(pts2.shape[0]))
        obs_r = np.sqrt(np.sum(pts ** 2, axis = 1))
        error = np.log10(np.abs((correct - est) / correct))
        plt.title(str(p))
        plt.plot(obs_r, error, '.')
        plt.show()


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
