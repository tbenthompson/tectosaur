import numpy as np
from dimension import dim, module
from tectosaur.util.test_decorators import slow

def test_bisects(dim):
    pts = np.random.rand(100,dim)
    t = module[dim].Octree(pts, pts, 1)
    pts = np.array(t.pts)
    for n in t.nodes:
        if n.is_leaf:
            continue
        idx_list = set(range(n.start, n.end))
        for child_i in range(2 ** dim):
            child_n = t.nodes[n.children[child_i]]
            child_idx_list = set(range(child_n.start, child_n.end))
            assert(child_idx_list.issubset(idx_list))
            idx_list -= child_idx_list
        assert(len(idx_list) == 0)

def test_contains_pts(dim):
    pts = np.random.rand(100,dim)
    t = module[dim].Octree(pts, pts, 1)
    pts = np.array(t.pts)
    for n in t.nodes:
        for i in range(n.start, n.end):
            assert(module[dim].in_box(n.bounds, pts[i,:].tolist()))

def test_height_depth(dim):
    pts = np.random.rand(100,dim)
    t = module[dim].Octree(pts, pts, 1)
    for n in t.nodes:
        if n.is_leaf:
            continue
        for c in range(2):
            assert(n.depth == t.nodes[n.children[c]].depth - 1)
        assert(n.height ==
            max([t.nodes[n.children[c]].height for c in range(2 ** dim)]) + 1)

def test_one_level(dim):
    pts = np.random.rand(dim, dim)
    t = module[dim].Octree(pts, pts, 4)
    assert(t.max_height == 0);
    assert(len(t.nodes) == 1);
    assert(t.root().is_leaf);
    assert(t.root().end - t.root().start);
    assert(t.root().depth == 0);
    assert(t.root().height == 0);

def test_orig_idxs(dim):
    pts = np.random.rand(1000,dim)
    t = module[dim].Octree(pts, pts, 50)
    np.testing.assert_almost_equal(np.array(t.pts), pts[np.array(t.orig_idxs), :])

def test_idx(dim):
    pts = np.random.rand(100,dim)
    t = module[dim].Octree(pts, pts, 1)
    for i, n in enumerate(t.nodes):
        assert(n.idx == i)

@slow
def test_law_of_large_numbers():
    n = 100000
    pts = np.random.rand(n, 3)
    t = module[3].Octree(pts, pts, 100);
    for i in range(8):
        child = t.nodes[t.root().children[i]]
        n_pts = child.end - child.start
        diff = np.abs(n_pts - (n / 8));
        assert(diff < (n / 16));
