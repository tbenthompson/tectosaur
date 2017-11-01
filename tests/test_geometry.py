from tectosaur.util.geometry import *

def test_longest_edge():
    assert(get_longest_edge(get_edge_lens(np.array([[0,0,0],[1,0,0],[0.5,0.5,0]]))) == 0)
    assert(get_longest_edge(get_edge_lens(np.array([[0,0,0],[0.5,0.5,0],[1,0,0]]))) == 2)

def test_vec_angle180():
    np.testing.assert_almost_equal(vec_angle(np.array([1,1]),np.array([-1,-1])), np.pi)

def test_tri_normal():
    tri = [[0,0,0],[1,0,0],[0,1,0]]
    np.testing.assert_almost_equal(tri_normal(tri), [0,0,1])

def test_tri_unit_normal():
    tri = [[0,0,0],[0,5,0],[0,0,5]]
    np.testing.assert_almost_equal(tri_normal(tri, normalize = True), [1,0,0])

def test_tri_area():
    np.testing.assert_almost_equal(tri_area(np.array([[0,0,0],[1,0,0],[0,1,0]])), 0.5)

def test_which_side_pt():
    tri = np.array([[0,0,0],[1,0,0],[0,1,0]])
    assert(which_side_point(tri, np.array([0,0,-1])) == Side.behind)
    assert(which_side_point(tri, np.array([0,0,1])) == Side.front)
    assert(which_side_point(tri, np.array([0,0,0])) == Side.intersect)
    assert(which_side_point(tri, np.array([0,0,1e-14])) == Side.intersect)

def test_tri_side():
    assert(tri_side([Side.front, Side.front, Side.front]) == Side.front);
    assert(tri_side([Side.intersect, Side.front, Side.front]) == Side.front);
    assert(tri_side([Side.intersect, Side.intersect, Side.front]) == Side.front);
    assert(tri_side([Side.behind, Side.intersect, Side.behind]) == Side.behind);
    assert(tri_side([Side.behind, Side.front, Side.behind]) == Side.intersect);

