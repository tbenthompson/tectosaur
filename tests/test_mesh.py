from tectosaur.mesh.refine import refine, selective_refine, refine_to_size
from tectosaur.mesh.modify import concat, flip_normals
from tectosaur.mesh.mesh_gen import make_rect
from tectosaur.util.geometry import tri_normal
import numpy as np

def test_remove_duplicates():
    surface1 = make_rect(2, 2, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    surface2 = make_rect(2, 2, [[0, 0, 0], [-1, 0, 0], [-1, 1, 0], [0, 1, 0]])
    m_f = concat(surface1, surface2)
    assert(m_f[0].shape[0] == 6)
    assert(m_f[1].shape[0] == 4)

def test_flip_normals():
    m = make_rect(2, 2, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    m_flip = flip_normals(m)
    for i in range(m[1].shape[0]):
        n1 = tri_normal(m[0][m[1][i,:]])
        n2 = tri_normal(m_flip[0][m_flip[1][i,:]])
        np.testing.assert_almost_equal(n1, -n2)

def test_refine():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    tris = np.array([[0, 1, 2], [3, 2, 1]])
    refined = refine((pts, tris))
    assert(refined[0].shape[0] == 9)

def test_selective_refine():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    tris = np.array([[0, 1, 2], [3, 2, 1]])
    refined = selective_refine((pts, tris), np.array([True, False]))
    assert(refined[0].shape[0] == 7)

def test_refine_to_size():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    tris = np.array([[0, 1, 2], [3, 2, 1]])
    m2, _ = refine_to_size((pts,tris), 0.2)
    np.testing.assert_almost_equal(m2[0][m2[1][1]], [[1,0,0],[0.5,0.5,0],[0.5,0,0]])
    assert(m2[1].shape[0] == 8)

def test_refine_to_size_with_field():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]])
    tris = np.array([[0, 1, 2], [3, 2, 1]])
    m2, fs = refine_to_size((pts,tris), 0.2, [pts[tris]])
    np.testing.assert_almost_equal(fs[0], m2[0][m2[1]])
