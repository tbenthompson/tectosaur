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

def test_octree_root():
    ctx = taskloaf.launch_local(1, taskloaf.Config())
    pts = np.random.rand(10, 3)
    o = fmm.make_octree(11, pts)
    hw = o.root.bounds.half_width
    c = o.root.bounds.center
    m1 = np.min(pts, axis = 0)
    m2 = np.max(pts, axis = 0)
    assert(np.all(pts <= np.array(c) + np.array(hw) * 1.00001))
    assert(np.all(pts >= np.array(c) - np.array(hw) * 1.00001))

def test_octree_split():
    ctx = taskloaf.launch_local(1, taskloaf.Config())
    pts = np.random.rand(20, 3)
    o = fmm.make_octree(2, pts)
    c = o.root.get_child(0)
    assert(fmm.n_total_children(o) == 20)
    #TODO:

@slow
def test_build_big():
    cfg = taskloaf.Config()
    cfg.print_stats = True
    cfg.interrupt_rate = 300
    ctx = taskloaf.launch_local(1, cfg)
    pts = np.random.rand(100000, 3)
    import time
    start = time.time()
    o = fmm.make_octree(1, pts, pts)
    assert(fmm.n_total_children(o) == pts.shape[0])
    print("Took: " + str(time.time() - start))

def foreach_node(o, f):
    remaining_nodes = [o.root]
    while len(remaining_nodes) > 0:
        node = remaining_nodes[-1]
        remaining_nodes.pop()
        f(node)
        if not node.is_leaf:
            for i in range(8):
                remaining_nodes.append(node.get_child(i))

def count_nodes(o):
    def increment(node):
        increment.i += 1
    increment.i = 0
    foreach_node(o, increment)
    return increment.i

def test_tree_traversal():
    ctx = taskloaf.launch_local(1, taskloaf.Config())
    pts = np.random.rand(10, 3)
    o = fmm.make_octree(2, pts)
    print("DONE")

def test_evaluate():
    ctx = taskloaf.launch_local(1)
    def gen_pts(n):
        return np.array([np.linspace(0, 1, n+2)[1:-1], np.zeros(n), np.zeros(n)]).T
    pts = gen_pts(100)
    o = fmm.make_octree(1, pts, pts)
    pts2 = gen_pts(101)
    inv_r = 1.0 / scipy.spatial.distance.cdist(pts2, pts)

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


def to_scipy(m, shape):
    return scipy.sparse.coo_matrix((m.vals, (m.rows, m.cols)), shape)

def get_nnz(fmm_mat):
    total_nnz = 0
    mats = []
    for mat in [fmm_mat.p2p, fmm_mat.p2m, fmm_mat.m2p, fmm_mat.m2m]:
        mat = to_scipy(mat)
        total_nnz += mat.nnz
        print(mat.nnz)
    print("total nnz: " + str(total_nnz))

def test_upward_traversal():
    ctx = taskloaf.launch_local(1)
    pts = np.random.rand(20000, 3)
    pts2 = np.random.rand(20000, 3)
    # pts = make_line_pts(1000)
    # pts2 = make_line_pts(999)

    import time
    start = time.time()
    o = fmm.make_octree(50, pts, pts)
    print(count_nodes(o))

    o2 = fmm.make_octree(50, pts2, pts2)
    surf = surrounding_surface_sphere(5)
    up = fmm.up_up_up(o, surf)
    fmm_mat = fmm.go_go_go(up, o2)

    n_src = pts.shape[0]
    n_obs = pts2.shape[0]
    n_multipoles = fmm_mat.n_m_dofs;

    p2p = to_scipy(fmm_mat.p2p, (n_obs, n_src)).tocsr()
    p2m = to_scipy(fmm_mat.p2m, (n_multipoles, n_src)).tocsr()
    m2p = to_scipy(fmm_mat.m2p, (n_obs, n_multipoles)).tocsr()
    m2m = to_scipy(fmm_mat.m2m, (n_multipoles, n_multipoles)).tocsc()
    print("TOOK: " + str(time.time() - start))

    input_vals = np.ones(pts.shape[0])

    start = time.time()
    m_lowest = p2m.dot(input_vals)
    m_all = scipy.sparse.linalg.spsolve(m2m, m_lowest)
    est = p2p.dot(input_vals) + m2p.dot(m_all)
    print("Eval took: " + str(time.time() - start))
    import ipdb; ipdb.set_trace()

    correct = (1.0 / scipy.spatial.distance.cdist(pts2, pts)).dot(np.ones(pts.shape[0]))
    error = np.mean(np.abs((est - correct) / correct))
    print(error)
    np.testing.assert_almost_equal(est, correct)

if __name__ == '__main__':
    # test_evaluate()
    # test_build_big()
    test_upward_traversal()
