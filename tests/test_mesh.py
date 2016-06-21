import tectosaur.mesh as mesh
from tectosaur.geometry import Side, which_side_point, tri_side
import numpy as np

def test_remove_duplicates():
    surface1 = mesh.rect_surface(2, 2, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    surface2 = mesh.rect_surface(2, 2, [[0, 0, 0], [-1, 0, 0], [-1, 1, 0], [0, 1, 0]])
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

