import sys
import numpy as np

import tectosaur.mesh as mesh
import pyximport; pyximport.install()
from tectosaur.octree import Octree

def tree_sum(node):
    if node.children is None:
        if node.pts is None:
            return 0
        return len(node.pts)
    else:
        return sum([tree_sum(node.children[i]) for i in range(8)])

def test_python_octree():
    pts = np.random.rand(100000, 3)
    import time
    start = time.time()
    o = Octree(1, pts)
    n_pts = tree_sum(o)
    assert(n_pts == pts.shape[0])
    print("Took: " + str(time.time() - start))

if __name__ == '__main__':
    test_python_octree()
