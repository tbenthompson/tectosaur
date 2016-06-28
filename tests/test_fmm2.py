import sys
import scipy.spatial
import numpy as np

import tectosaur.mesh as mesh

from test_decorators import slow

import taskloaf
import cppimport
fmm = cppimport.imp("tectosaur.fmm").fmm

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


def test_upward_traversal():
    ctx = taskloaf.launch_local(4)
    pts = np.random.rand(100000, 3)
    import time
    start = time.time()
    o = fmm.make_octree(1, pts, pts)

    def increment(n):
        increment.i += 1
    increment.i = 0

    foreach_node(o, increment)
    print(increment.i)
    print(time.time() -start)
    # surf = surrounding_surface_sphere(8)
    # up = fmm.up_up_up(o, surf)

if __name__ == '__main__':
    # test_evaluate()
    # test_build_big()
    test_upward_traversal()
