import numpy as np
from tectosaur.util.geometry import tri_normal

def get_slip_vectors(tri, vertical = [0, 0, 1]):
    n = tri_normal(tri, normalize = True)
    is_normal_vertical = n.dot(vertical) >= 1.0
    if is_normal_vertical: # this means the fault plane is horizontal, so there is no "strike" and "dip"
        raise Exception("fault plane is horizontal. strike and dip make no sense")
    v1 = np.cross(n, vertical)
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    v2 /= np.linalg.norm(v2)
    return v1, v2

def test_slip_vec_easy():
    v1, v2 = get_slip_vectors(np.array([[0,0,0],[1,0,0],[0,1,0]]))

def test_slip_vec_hard():
    v1, v2 = get_slip_vectors(np.array([[0,0,0],[0,1,0],[0,0,1]]))
    np.testing.assert_almost_equal(v1, [0,0,1])
    np.testing.assert_almost_equal(v2, [0,-1,0])

def test_slip_vec_harder():
    for i in range(10):
        # random triangles should still follow these rules:
        # vecs should be perpindicular to each other and the normal
        # and be normalized to unit length
        tri = np.random.rand(3,3)
        v1, v2 = get_slip_vectors(tri)
        n = tri_normal(tri, normalize = True)
        np.testing.assert_almost_equal(np.linalg.norm(v1), 1.0)
        np.testing.assert_almost_equal(np.linalg.norm(v2), 1.0)
        np.testing.assert_almost_equal(v1.dot(v2), 0.0)
        np.testing.assert_almost_equal(v1.dot(n), 0.0)
        np.testing.assert_almost_equal(v2.dot(n), 0.0)