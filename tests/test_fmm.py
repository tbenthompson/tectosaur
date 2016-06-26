import sys
import numpy as np

import tectosaur.mesh as mesh

import taskloaf
import cppimport
fmm = cppimport.imp("tectosaur.fmm").fmm

class Octree:
    def __init__(self, max_pts, pts):
        if pts.shape[0] == 0:
            self.pts = None
            self.children = None
            return

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
                        self.build_child(s1, s2, s3, comp, max_pts, pts)

    def build_child(self, s1, s2, s3, comp, max_pts, pts):
        cond = np.logical_and(
            np.logical_and(
                comp[0][s1], comp[1][s2]
            ), comp[2][s3]
        )
        child_pts = pts[cond]
        self.children.append(
            taskloaf.task(lambda: Octree(max_pts, child_pts))
        )

def reduce(lst, fnc):
    if len(lst) == 1:
        return lst[0]
    else:
        middle_idx = len(lst) // 2
        left = reduce(lst[:middle_idx], fnc)
        right = reduce(lst[middle_idx:], fnc)
        return left.then(lambda a: right.then(lambda b: fnc(a, b))).unwrap()

def tree_sum(o):
    def tree_sum_helper(node):
        if node.children is None:
            if node.pts is None:
                return taskloaf.ready(0)
            return taskloaf.ready(len(node.pts))
        else:
            sums = [node.children[i].then(tree_sum_helper).unwrap() for i in range(8)]
            return reduce(sums, lambda x, y: x + y)
    return (o.then(tree_sum_helper).unwrap())

def test_python_octree():
    # from mpi4py import MPI
    ctx = taskloaf.launch_local(1)
    pts = np.random.rand(5000000, 3)
    import time
    start = time.time()
    o = taskloaf.task(lambda: Octree(5000, pts))
    n_pts = tree_sum(o).get()
    assert(n_pts == pts.shape[0])
    print("Took: " + str(time.time() - start))

#
# def test_fmm_cpp():
#     fmm.run_tests([])
#
# def test_octree_root():
#     ctx = taskloaf.launch_local(1)
#     pts = np.random.rand(10, 3)
#     o = fmm.Octree(11, pts)
#     hw = o.root.bounds.half_width
#     c = o.root.bounds.center
#     assert(np.all(pts <= np.array(c) + np.array(hw)))
#     assert(np.all(pts >= np.array(c) - np.array(hw)))
#
# def test_octree_split():
#     ctx = taskloaf.launch_local(1)
#     pts = np.random.rand(10, 3)
#     o = fmm.Octree(1, pts)
#     #TODO:
#     # hw = o.root.bounds.half_width
#     # c = o.root.bounds.center
#     # print(np.prod(hw))
#
# def test_build_big():
#     ctx = taskloaf.launch_local(1)
#     pts = np.random.rand(50000000, 3)
#     import time
#     start = time.time()
#     o = fmm.Octree(50, pts)
#     print(fmm.tree_sum(o))
#     print("Took: " + str(time.time() - start))

if __name__ == '__main__':
    test_python_octree()
    # test_build_big()
    # fmm.run_tests(sys.argv)
