import numpy as np

import tectosaur.mesh.mesh_gen as mesh_gen
import tectosaur.mesh.find_nearfield as find_nearfield
import tectosaur.mesh.adjacency as adjacency

from tectosaur.util.test_decorators import golden_master,flatten

from tectosaur.util.logging import setup_logger
setup_logger(__name__)

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

@golden_master()
def test_find_adjacency(request):
    m = mesh_gen.make_sphere([0,0,0], 1.0, 3)
    close_pairs = find_nearfield.find_close_or_touching(m[0], m[1], 1.25)
    close, va, ea = find_nearfield.split_adjacent_close(close_pairs, m[1])
    # va, ea = adjacency.find_adjacents(m)
    va = np.array(va)
    ea = np.array(ea)
    all = np.zeros((va.shape[0] + ea.shape[0], 6),)
    all[:va.shape[0],:4] = va
    all[va.shape[0]:] = ea
    sorted_idxs = np.lexsort([all[:,1], all[:,0]], axis = 0)
    all_sorted = all[sorted_idxs,:]
    return all_sorted

def benchmark_find_nearfield():
    corners = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    nx = ny = 707
    pts, tris = mesh_gen.make_rect(nx, ny, corners)
    print('n_tris: ' + str(tris.shape[0]))
    # va, ea = adjacency.find_adjacents(tris)
    near_pairs = find_nearfield.find_close_or_touching(pts, tris, 1.25)

def benchmark_find_adjacency():
    from tectosaur.mesh.mesh_gen import make_rect
    from tectosaur.util.timer import Timer
    L = 10
    nx = ny = 2 ** L
    t = Timer()
    m = make_rect(nx, ny, [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])
    t.report('make')
    close_pairs = find_nearfield.find_close_or_touching(m[0], m[1], 1.25)
    t.report('close or touching')
    close, va, ea = find_nearfield.split_adjacent_close(close_pairs, m[1])
    t.report('find adj new')
    print(m[1].shape[0])
    va, ea = adjacency.find_adjacents(m[1])
    t.report('find adj')
    # va, ea = find_adjacents_old(m[1])
    # t.report('find adj old')

if __name__ == "__main__":
    benchmark_find_adjacency()
    # benchmark_find_nearfield()
