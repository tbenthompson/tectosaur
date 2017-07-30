import numpy as np

import tectosaur.mesh.mesh_gen as mesh_gen
import tectosaur.mesh.find_nearfield as find_nearfield
import tectosaur.mesh.adjacency as adjacency

from tectosaur.util.test_decorators import golden_master,flatten

def test_nearfield():
    corners = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    pts, tris = mesh_gen.make_rect(3,3 ,corners)
    assert(tris.shape[0] == 8)
    va, ea = adjacency.find_adjacents(tris)
    near_pairs = find_nearfield.find_nearfield(pts, tris, va, ea, 2.5)
    check_for = [
        (0, 5), (0, 6), (0, 3), (1, 7), (2, 7), (3, 0),
        (4, 7), (5, 0), (6, 0), (7, 2), (7, 1), (7, 4)
    ]
    assert(len(near_pairs) == len(check_for))
    for pair in check_for:
        assert(pair in near_pairs)

@golden_master()
def test_close_or_touching(request):
    n = 20
    pts, tris = mesh_gen.make_rect(n, n, [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    near_pairs = find_nearfield.find_close_or_touching(pts, tris, 1.25)
    return np.sort(near_pairs, axis = 0)

@golden_master()
def test_find_nearfield_real(request):
    n = 20
    pts, tris = mesh_gen.make_rect(n, n, [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    va, ea = adjacency.find_adjacents(tris)
    return list(flatten([
        sorted(x) for x in find_nearfield.find_nearfield(pts, tris, va, ea, 1.25)
    ]))

def benchmark_find_nearfield():
    corners = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    nx = ny = 707
    pts, tris = mesh_gen.make_rect(nx, ny, corners)
    print('n_tris: ' + str(tris.shape[0]))
    # va, ea = adjacency.find_adjacents(tris)
    near_pairs = find_nearfield.find_close_or_touching(pts, tris, 1.25)

if __name__ == '__main__':
    benchmark_find_nearfield()
