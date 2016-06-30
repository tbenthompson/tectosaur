import sys
import scipy.spatial
import scipy.sparse
import scipy.sparse.linalg
import numpy as np

import tectosaur.mesh as mesh

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
    for n in kdtree.nodes:
        for d in range(3):
            assert(np.all(pts[n.start:n.end,d] <=
                n.bounds.center[d] + n.bounds.half_width[d] * 1.0001))
            assert(np.all(pts[n.start:n.end,d] >=
                n.bounds.center[d] - n.bounds.half_width[d] * 1.0001))

def test_p2m():
    pts1 = np.random.rand(100,3)
    pts2 = np.random.rand(100,3)
    pts2[:,0] += 20
    kd1 = fmm.KDTree(pts1, pts1, 101)
    kd2 = fmm.KDTree(pts2, pts2, 1)
    surf = surrounding_surface_sphere(10)
    fmm_mat = fmm.fmmmmmmm(kd1, kd2, fmm.FMMConfig(0.3, surf, "one"))
    m_lowest = to_scipy(fmm_mat.p2m).dot(np.ones(100))
    for n in kd2.nodes:
        if not n.is_leaf:
            continue
        start_mdof = n.idx * surf.shape[0]
        end_mdof = (n.idx + 1) * surf.shape[0]
        assert(int(np.sum(m_lowest[start_mdof:end_mdof])) == (n.end - n.start))

def test_m2m_m2p():
    pts1 = np.random.rand(100,3)
    pts2 = np.random.rand(105,3)
    pts2[:,0] += 1.7
    kd1 = fmm.KDTree(pts1, pts1, 2)
    kd2 = fmm.KDTree(pts2, pts2, 2)
    surf = surrounding_surface_sphere(10)
    fmm_mat = fmm.fmmmmmmm(kd1, kd2, fmm.FMMConfig(0.3, surf, "one"))

    assert(to_scipy(fmm_mat.p2p).nnz == 0)

    m_lowest = to_scipy(fmm_mat.p2m).dot(np.ones(pts2.shape[0]))
    m_all = scipy.sparse.linalg.spsolve(to_scipy(fmm_mat.m2m), -m_lowest)
    assert(m_all.shape[0] == len(kd2.nodes) * surf.shape[0])
    n_surf = surf.shape[0]
    for n in kd2.nodes:
        start_mdof = n.idx * surf.shape[0]
        end_mdof = (n.idx + 1) * surf.shape[0]
        m_strength = int(np.round(np.sum(m_all[start_mdof:end_mdof])))
        n_pts = n.end - n.start
        assert(m_strength == n_pts)

    m2p = to_scipy(fmm_mat.m2p, (pts1.shape[0], m_all.shape[0]))
    result = m2p.dot(m_all)
    assert(np.all(np.abs(result - pts2.shape[0]) < 1e-3))


def run_full(n, kernel):
    pts = np.random.rand(n, 3)
    pts2 = np.random.rand(n, 3)
    # pts = make_line_pts(5000)
    # pts2 = make_line_pts(4999)

    import time
    start = time.time()
    kd = fmm.KDTree(pts, pts, 10)
    kd2 = fmm.KDTree(pts2, pts2, 10)
    surf = surrounding_surface_sphere(40)
    fmm_mat = fmm.fmmmmmmm(kd2, kd, fmm.FMMConfig(0.3, surf, kernel))
    print("Evaluate: " + str(time.time() - start))

    n_src = pts.shape[0]
    n_obs = pts2.shape[0]

    start = time.time()
    p2p = to_scipy(fmm_mat.p2p)
    p2m = to_scipy(fmm_mat.p2m)
    m2m = to_scipy(fmm_mat.m2m)
    m2p = to_scipy(fmm_mat.m2p, (pts.shape[0], m2m.shape[0]))
    print("copy to scipy: " + str(time.time() - start))



    input_vals = np.ones(pts.shape[0])

    start = time.time()
    m_lowest = p2m.dot(input_vals)
    m_all = scipy.sparse.linalg.spsolve(m2m, -m_lowest)
    m2p_comp = m2p.dot(m_all)
    p2p_comp = p2p.dot(input_vals)
    est = p2p_comp + m2p_comp
    print("Eval took: " + str(time.time() - start))
    return pts, pts2, est

def test_ones():
    pts, pts2, est = run_full(5000, "one")
    assert(np.all(np.abs(est - 5000) < 1e-3))

def test_inv_r():
    pts, pts2, est = run_full(5000, "invr")
    correct = (1.0 / (scipy.spatial.distance.cdist(pts2, pts))).dot(np.ones(pts.shape[0]))
    error = np.mean((est - correct) ** 2)
    print(error)
    np.testing.assert_almost_equal(est, correct, -1)

@slow
def test_build_big():
    pts = np.random.rand(1000000, 3)
    import time
    start = time.time()
    kdtree = fmm.KDTree(pts, pts, 1)
    print("KDTree took: " + str(time.time() - start))

if __name__ == '__main__':
    # test_evaluate()
    test_build_big()
    test_upward_traversal()
