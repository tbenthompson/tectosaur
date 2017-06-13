import tectosaur.mesh as mesh
from tectosaur.geometry import Side, which_side_point, tri_side
import numpy as np

def test_remove_duplicates():
    surface1 = mesh.make_rect(2, 2, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    surface2 = mesh.make_rect(2, 2, [[0, 0, 0], [-1, 0, 0], [-1, 1, 0], [0, 1, 0]])
    m_f = mesh.concat(surface1, surface2)
    assert(m_f[0].shape[0] == 6)
    assert(m_f[1].shape[0] == 4)

def test_which_side_pt():
    tri = np.array([[0,0,0],[1,0,0],[0,1,0]])
    assert(which_side_point(tri, np.array([0,0,-1])) == Side.behind)
    assert(which_side_point(tri, np.array([0,0,1])) == Side.front)
    assert(which_side_point(tri, np.array([0,0,0])) == Side.intersect)

def test_tri_side():
    assert(tri_side([Side.front, Side.front, Side.front]) == Side.front);
    assert(tri_side([Side.intersect, Side.front, Side.front]) == Side.front);
    assert(tri_side([Side.intersect, Side.intersect, Side.front]) == Side.front);
    assert(tri_side([Side.behind, Side.intersect, Side.behind]) == Side.behind);

def test_refine():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    tris = np.array([[0, 1, 2], [3, 2, 1]])
    refined = mesh.refine((pts, tris))
    assert(refined[0].shape[0] == 9)

def test_selective_refine():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    tris = np.array([[0, 1, 2], [3, 2, 1]])
    refined = mesh.selective_refine((pts, tris), np.array([True, False]))
    assert(refined[0].shape[0] == 7)

def test_refine_to_size():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    tris = np.array([[0, 1, 2], [3, 2, 1]])
    m2, _ = mesh.refine_to_size((pts,tris), 0.2)
    np.testing.assert_almost_equal(m2[0][m2[1][1]], [[1,0,0],[0.5,0.5,0],[0.5,0,0]])
    assert(m2[1].shape[0] == 8)

def test_refine_to_size_with_field():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    tris = np.array([[0, 1, 2], [3, 2, 1]])
    m2, fs = mesh.refine_to_size((pts,tris), 0.2, [pts[tris]])
    np.testing.assert_almost_equal(fs[0], m2[0][m2[1]])
