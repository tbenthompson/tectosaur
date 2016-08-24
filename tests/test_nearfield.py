import tectosaur.find_nearfield
import tectosaur.mesh
import tectosaur.adjacency

def test_nearfield():
    corners = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    pts, tris = tectosaur.mesh.rect_surface(3,3 ,corners)
    assert(tris.shape[0] == 8)
    va, ea = tectosaur.adjacency.find_adjacents(tris)
    near_pairs = tectosaur.find_nearfield.find_nearfield(pts, tris, va, ea, 2.5)
    check_for = [
        (0, 5), (0, 6), (0, 3), (1, 7), (2, 7), (3, 0),
        (4, 7), (5, 0), (6, 0), (7, 2), (7, 1), (7, 4)
    ]
    assert(len(near_pairs) == len(check_for))
    for pair in check_for:
        assert(pair in near_pairs)
