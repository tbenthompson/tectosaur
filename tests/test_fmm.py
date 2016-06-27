import sys
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
    o = fmm.Octree(11, pts)
    hw = o.root.bounds.half_width
    c = o.root.bounds.center
    m1 = np.min(pts, axis = 0)
    m2 = np.max(pts, axis = 0)
    assert(np.all(pts <= np.array(c) + np.array(hw) * 1.00001))
    assert(np.all(pts >= np.array(c) - np.array(hw) * 1.00001))

def test_octree_split():
    ctx = taskloaf.launch_local(1, taskloaf.Config())
    pts = np.random.rand(20, 3)
    o = fmm.Octree(2, pts)
    c = o.root.get_child(0)
    assert(fmm.n_total_children(o) == 20)
    #TODO:

@slow
def test_build_big():
    ctx = taskloaf.launch_local(6, taskloaf.Config())
    pts = np.random.rand(27000, 3)
    import time
    start = time.time()
    o = fmm.Octree(10, pts)
    assert(fmm.n_total_children(o) == pts.shape[0])
    print("Took: " + str(time.time() - start))

def test_tree_traversal():
    ctx = taskloaf.launch_local(1, taskloaf.Config())
    pts = np.random.rand(10, 3)
    o = fmm.Octree(2, pts)
    remaining_nodes = [o.root]
    while len(remaining_nodes) > 0:
        node = remaining_nodes[-1]
        remaining_nodes.pop()
        if node.is_leaf:
            print(np.array(node))
        else:
            for i in range(8):
                remaining_nodes.append(node.get_child(i))
    print("DONE")

def dual_tree(node1, node2):
    c1 = node1.bounds.center
    c2 = node2.bounds.center
    # dist =

def test_upward_traversal():
    ctx = taskloaf.launch_local(1)
    pts = np.random.rand(27000, 3)
    o = fmm.Octree(100, pts)
    up = fmm.up_up_up(o);


if __name__ == '__main__':
    pass
