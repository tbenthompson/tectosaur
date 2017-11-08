from tectosaur.util.geometry import *

def test_internal_angles():
    angles = triangle_internal_angles([[0,0,0],[1,0,0],[0,1,0]])
    np.testing.assert_almost_equal(angles, [np.pi / 2, np.pi / 4, np.pi / 4])

def test_longest_edge():
    assert(get_longest_edge(get_edge_lens([[0,0,0],[1,0,0],[0.5,0.5,0]])) == 0)
    assert(get_longest_edge(get_edge_lens([[0,0,0],[0.5,0.5,0],[1,0,0]])) == 2)

def test_vec_angle180():
    np.testing.assert_almost_equal(vec_angle([1,1,0],[-1,-1,0]), np.pi)

def test_tri_normal():
    tri = [[0,0,0],[1,0,0],[0,1,0]]
    np.testing.assert_almost_equal(tri_normal(tri), [0,0,1])

def test_tri_unit_normal():
    tri = [[0,0,0],[0,5,0],[0,0,5]]
    np.testing.assert_almost_equal(tri_normal(tri, normalize = True), [1,0,0])

def test_tri_area():
    np.testing.assert_almost_equal(tri_area(np.array([[0,0,0],[1,0,0],[0,1,0]])), 0.5)
