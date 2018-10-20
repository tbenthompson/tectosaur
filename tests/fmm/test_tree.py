import numpy as np
from dimension import dim
from tectosaur.fmm.cfg import get_dim_module
from tectosaur.util.test_decorators import slow
import pytest

@pytest.fixture(params = ['kd', 'oct'])
def tree_type(request):
    return request.param

def make_tree(tree_type, pts, Rs, n_per_cell):
    dim = pts.shape[1]
    if tree_type == 'kd':
        return get_dim_module(dim).kdtree.Tree.build(pts, Rs, n_per_cell)
    elif tree_type == 'oct':
        return get_dim_module(dim).octree.Tree.build(pts, Rs, n_per_cell)

def simple_setup(n, tree_type, dim):
    pts = np.random.rand(n, dim)
    Rs = np.random.rand(n) * 0.01
    t = make_tree(tree_type, pts, Rs, 1)
    return pts, Rs, t

def test_bisects(tree_type, dim):
    pts, Rs, t = simple_setup(100, tree_type, dim)
    pts = np.array([b.center for b in t.balls])
    for n in t.nodes:
        if n.is_leaf:
            continue
        idx_list = set(range(n.start, n.end))
        for child_i in range(t.split):
            child_n = t.nodes[n.children[child_i]]
            child_idx_list = set(range(child_n.start, child_n.end))
            assert(child_idx_list.issubset(idx_list))
            idx_list -= child_idx_list
        assert(len(idx_list) == 0)

def test_contains_pts(tree_type, dim):
    pts, Rs, t = simple_setup(100, tree_type, dim)
    pts = np.array([b.center for b in t.balls])
    for n in t.nodes:
        for i in range(n.start, n.end):
            dist = np.sqrt(np.sum((n.bounds.center - pts[i,:]) ** 2))
            assert(dist <= n.bounds.R)

def test_height_depth(tree_type, dim):
    pts, Rs, t = simple_setup(100, tree_type, dim)
    for n in t.nodes:
        if n.is_leaf:
            continue
        for c in range(t.split):
            assert(n.depth == t.nodes[n.children[c]].depth - 1)
        assert(n.height ==
            max([t.nodes[n.children[c]].height for c in range(t.split)]) + 1)

def test_one_level(tree_type, dim):
    pts = np.random.rand(dim, dim)
    t = make_tree(tree_type, pts, np.full(pts.shape[0], 0.01), dim + 1)
    assert(t.max_height == 0);
    assert(len(t.nodes) == 1);
    assert(t.root().is_leaf);
    assert(t.root().end - t.root().start);
    assert(t.root().depth == 0);
    assert(t.root().height == 0);

def test_orig_idxs(tree_type, dim):
    pts = np.random.rand(1000, dim)
    Rs = np.random.rand(1000) * 0.01
    t = make_tree(tree_type, pts, Rs, 50)
    pts_new = np.array([b.center for b in t.balls])
    np.testing.assert_almost_equal(pts_new, pts[np.array(t.orig_idxs), :])

def test_idx(tree_type, dim):
    pts, Rs, t = simple_setup(100, tree_type, dim)
    for i, n in enumerate(t.nodes):
        assert(n.idx == i)

def test_law_of_large_numbers():
    n = 10000
    pts = np.random.rand(n, 3)
    Rs = np.random.rand(n) * 0.01
    t = get_dim_module(3).octree.Tree.build(pts, Rs, 100);
    for i in range(8):
        child = t.nodes[t.root().children[i]]
        n_pts = child.end - child.start
        diff = np.abs(n_pts - (n / 8));
        assert(diff < (n / 16));
