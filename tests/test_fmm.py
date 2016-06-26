import sys
import numpy as np

import tectosaur.mesh as mesh

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
    b = o.root.get_child(0).bounds.center
    print(b)
    #TODO:
    # hw = o.root.bounds.half_width
    # c = o.root.bounds.center
    # print(np.prod(hw))

def test_build_big():
    ctx = taskloaf.launch_local(6, taskloaf.Config())
    pts = np.random.rand(1000000, 3)
    import time
    start = time.time()
    o = fmm.Octree(20, pts)
    print(fmm.tree_sum(o))
    print("Took: " + str(time.time() - start))

if __name__ == '__main__':
    # test_octree_split()
    test_build_big()
    # fmm.run_tests(sys.argv)
