import numpy as np

from tectosaur.adjacency import *

tris = [[0, 1, 2], [2, 1, 3], [0, 4, 5]]
def test_find_adjacents():
    va, ea = find_adjacents(tris)
    assert(len(va) == 2)
    assert(len(ea) == 2)
    assert(va[0] == (0, 2, [(0, 0)]))
    assert(va[1] == (2, 0, [(0, 0)]))
    assert(ea[0] == (0, 1, [(1, 1), (2, 0)]))
    assert(ea[1] == (1, 0, [(0, 2), (1, 1)]))

def test_rotate_tri():
    assert(rotate_tri(1) == [1, 2, 0])
    assert(rotate_tri(2) == [2, 0, 1])

def test_vert_adj_prep():
    tris = np.array([[0, 1, 2], [1, 3, 4], [6, 7, 2]])
    va, ea = find_adjacents(tris)
    result = vert_adj_prep(tris, va)
    assert(np.all(result[3][:, 0] == result[4][:, 0]))

def test_edge_adj_prep():
    tris = np.array([[0, 1, 2], [1, 3, 2]])
    va, ea = find_adjacents(tris)
    result = edge_adj_prep(tris, ea)

    tris = np.array([[0, 1, 2], [2, 1, 3]])
    va, ea = find_adjacents(tris)
    result = edge_adj_prep(tris, ea)

def test_find_free_edges():
    tris = np.array([[0,1,2],[2,1,3]])
    free_es = find_free_edges(tris)
    assert(len(free_es) == 4)
    for e in [(0,0), (0,2), (1,1), (1,2)]:
        assert(e in free_es)
