import numpy as np
from dimension import dim
from tectosaur.fmm.fmm_wrapper import module
from tectosaur.util.test_decorators import slow
import pytest

@pytest.fixture(params = ['kd', 'oct'])
def tree_type(request):
    return request.param

def make_tree(tree_type, pts, n_per_cell):
    dim = pts.shape[1]
    if tree_type == 'kd':
        return module[dim].kdtree.Tree(pts, n_per_cell)
    elif tree_type == 'oct':
        return module[dim].octree.Tree(pts, n_per_cell)

def test_bisects(tree_type, dim):
    pts = np.random.rand(100,dim)
    t = make_tree(tree_type, pts, 1)
    pts = np.array(t.pts)
    for n in t.nodes:
        if n.is_leaf:
            continue
        idx_list = set(range(n.start, n.end))
        for child_i in range(t.n_split()):
            child_n = t.nodes[n.children[child_i]]
            child_idx_list = set(range(child_n.start, child_n.end))
            assert(child_idx_list.issubset(idx_list))
            idx_list -= child_idx_list
        assert(len(idx_list) == 0)

def test_contains_pts(tree_type, dim):
    pts = np.random.rand(100,dim)
    t = make_tree(tree_type, pts, 1)
    pts = np.array(t.pts)
    for n in t.nodes:
        for i in range(n.start, n.end):
            dist = np.sqrt(np.sum((n.bounds.center - pts[i,:]) ** 2))
            assert(dist <= n.bounds.R)

def test_height_depth(tree_type, dim):
    pts = np.random.rand(100,dim)
    t = make_tree(tree_type, pts, 1)
    for n in t.nodes:
        if n.is_leaf:
            continue
        for c in range(t.n_split()):
            assert(n.depth == t.nodes[n.children[c]].depth - 1)
        assert(n.height ==
            max([t.nodes[n.children[c]].height for c in range(t.n_split())]) + 1)

def test_one_level(tree_type, dim):
    pts = np.random.rand(dim, dim)
    t = make_tree(tree_type, pts, dim + 1)
    assert(t.max_height == 0);
    assert(len(t.nodes) == 1);
    assert(t.root().is_leaf);
    assert(t.root().end - t.root().start);
    assert(t.root().depth == 0);
    assert(t.root().height == 0);

def test_orig_idxs(tree_type, dim):
    pts = np.random.rand(1000,dim)
    t = make_tree(tree_type, pts, 50)
    np.testing.assert_almost_equal(np.array(t.pts), pts[np.array(t.orig_idxs), :])

def test_idx(tree_type, dim):
    pts = np.random.rand(100,dim)
    t = make_tree(tree_type, pts, 1)
    for i, n in enumerate(t.nodes):
        assert(n.idx == i)

def test_law_of_large_numbers():
    n = 10000
    pts = np.random.rand(n, 3)
    t = module[3].octree.Tree(pts, 100);
    for i in range(8):
        child = t.nodes[t.root().children[i]]
        n_pts = child.end - child.start
        diff = np.abs(n_pts - (n / 8));
        assert(diff < (n / 16));
