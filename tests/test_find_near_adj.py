import numpy as np

import tectosaur.mesh.mesh_gen as mesh_gen
from tectosaur.mesh.find_near_adj import *

from tectosaur.util.test_decorators import golden_master,flatten

from tectosaur.util.logging import setup_logger
logger = setup_logger(__name__)

def find_adjacents(tris):
    pts = np.random.rand(np.max(tris) + 1, 3)
    close_pairs = find_close_or_touching(pts, tris, 1.0)
    close, va, ea = split_adjacent_close(close_pairs, tris)
    return va, ea

tris = np.array([[0, 1, 2], [2, 1, 3], [0, 4, 5]])
def test_find_adjacents():
    va, ea = find_adjacents(tris)
    assert(va.size == 8)
    assert(ea.size == 12)
    assert(np.all(va.flatten() == (0, 2, 0, 0, 2, 0, 0, 0)))
    assert(np.all(ea.flatten() == (0, 1, 1, 1, 2, 0, 1, 0, 0, 2, 1, 1)))

def test_rotate_tri():
    assert(rotate_tri(1) == [1, 2, 0])
    assert(rotate_tri(2) == [2, 0, 1])

def test_vert_adj_prep():
    tris = np.array([[0, 1, 2], [1, 3, 4], [6, 7, 2]])
    va, ea = find_adjacents(tris)
    result = vert_adj_prep(tris, va.reshape((-1, 4)))
    assert(np.all(result[3][:, 0] == result[4][:, 0]))

def test_edge_adj_prep():
    tris = np.array([[0, 1, 2], [1, 3, 2]])
    va, ea = find_adjacents(tris)
    result = edge_adj_prep(tris, ea.reshape((-1, 6)))

    tris = np.array([[0, 1, 2], [2, 1, 3]])
    va, ea = find_adjacents(tris)
    result = edge_adj_prep(tris, ea.reshape((-1, 6)))

def test_nearfield():
    corners = [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]
    pts, tris = mesh_gen.make_rect(3,3 ,corners)
    assert(tris.shape[0] == 8)
    close_pairs = find_close_or_touching(pts, tris, 1.0)
    close, va, ea = split_adjacent_close(close_pairs, tris)
    check_for = [
        (0, 5), (0, 6), (0, 3), (1, 7), (2, 7), (3, 0),
        (4, 7), (5, 0), (6, 0), (7, 2), (7, 1), (7, 4)
    ]
    assert(len(close) == len(check_for))
    for pair in check_for:
        assert(pair in close)

@golden_master()
def test_close_or_touching(request):
    n = 20
    pts, tris = mesh_gen.make_rect(n, n, [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    near_pairs = find_close_or_touching(pts, tris, 1.25)
    return np.sort(near_pairs, axis = 0)

@golden_master()
def test_find_nearfield_real(request):
    n = 20
    pts, tris = mesh_gen.make_rect(n, n, [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    va, ea = find_adjacents(tris)
    close_pairs = find_close_or_touching(pts, tris, 1.25)
    close, va, ea = split_adjacent_close(close_pairs, tris)
    return close[np.lexsort([close[:,1],close[:,0]], axis = 0)].flatten()

@golden_master()
def test_find_adjacency(request):
    m = mesh_gen.make_sphere([0,0,0], 1.0, 3)
    close_pairs = find_close_or_touching(m[0], m[1], 1.25)
    close, va, ea = split_adjacent_close(close_pairs, m[1])
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
    near_pairs = find_close_or_touching(pts, tris, 1.25)

def benchmark_find_adjacency():
    from tectosaur.mesh.mesh_gen import make_rect
    from tectosaur.util.timer import Timer
    L = 10
    nx = ny = int(2 ** L / np.sqrt(2))
    t = Timer()
    m = make_rect(nx, ny, [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])
    logger.debug('n_tris: ' + str(m[1].shape[0]))
    t.report('make')
    close_pairs = find_close_or_touching(m[0], m[1], 1.25)
    t.report('close or touching')
    close, va, ea = split_adjacent_close(close_pairs, m[1])
    t.report('find adj')

if __name__ == "__main__":
    benchmark_find_adjacency()
    # benchmark_find_nearfield()