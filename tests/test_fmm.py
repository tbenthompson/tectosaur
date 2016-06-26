import sys
import numpy as np

import tectosaur.mesh as mesh

import taskloaf
import cppimport
fmm = cppimport.imp("tectosaur.fmm").fmm

def test_fmm_cpp():
    fmm.run_tests([])

def test_octree_root():
    pts = np.random.rand(10, 3)
    o = fmm.Octree(11, pts)
    hw = o.root.bounds.half_width
    c = o.root.bounds.center
    assert(np.all(pts <= np.array(c) + np.array(hw)))
    assert(np.all(pts >= np.array(c) - np.array(hw)))

def test_octree_split():
    pts = np.random.rand(10, 3)
    o = fmm.Octree(1, pts)
    #TODO:
    # hw = o.root.bounds.half_width
    # c = o.root.bounds.center
    # print(np.prod(hw))

class Octree:
    def __init__(self, max_pts, pts):
        min_corner = np.min(pts, axis = 0)
        max_corner = np.max(pts, axis = 0)
        self.center = (min_corner + max_corner) / 2
        self.half_width = (min_corner - max_corner) / 2

        if pts.shape[0] < max_pts:
            self.pts = pts
            self.children = None
        else:
            self.pts = None
            self.children = []
            comp = [
                [pts[:,d] < self.center[d], pts[:,d] >= self.center[d]]
                for d in range(3)
            ]
            for s1 in range(2):
                for s2 in range(2):
                    for s3 in range(2):
                        cond = np.logical_and(np.logical_and(comp[0][s1],comp[1][s2]), comp[2][s3])
                        self.children.append(Octree(max_pts, pts[cond]))

def test_python_octree():
    pts = np.random.rand(5000000, 3)
    import time
    start = time.time()
    o = Octree(50, pts)
    print("Took: " + str(time.time() - start))

def test_build_big():
    ctx = taskloaf.launch_local(6)
    pts = np.random.rand(50000000, 3)
    import time
    start = time.time()
    o = fmm.Octree(50, pts)
    print(fmm.tree_sum(o))
    print("Took: " + str(time.time() - start))

if __name__ == '__main__':
    test_build_big()
    # test_python_octree()
    # fmm.run_tests(sys.argv)
