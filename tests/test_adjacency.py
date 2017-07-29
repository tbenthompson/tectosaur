import numpy as np

from tectosaur.mesh.adjacency import *
from tectosaur.mesh.mesh_gen import make_sphere, make_rect
from tectosaur.util.test_decorators import slow,golden_master

tris = [[0, 1, 2], [2, 1, 3], [0, 4, 5]]
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

def test_find_free_edges():
    tris = np.array([[0,1,2],[2,1,3]])
    free_es = find_free_edges(tris)
    assert(len(free_es) == 4)
    for e in [(0,0), (0,2), (1,1), (1,2)]:
        assert(e in free_es)

def flatten_list_helper(a):
    a = list(a)
    out = []
    if type(a) is list:
        out += a
    else:
        out.append(a)
    return out

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

@golden_master()
def test_find_adjacency(request):
    va, ea = find_adjacents(make_sphere([0,0,0], 1.0, 3)[1])
    return list(flatten(va.tolist() + ea.tolist()))

@slow
def test_bench_find_adjacency():
    from tectosaur.mesh.mesh_gen import make_rect
    from tectosaur.util.timer import Timer
    L = 10
    nx = ny = 2 ** L
    t = Timer()
    m = make_rect(nx, ny, [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]])
    t.report('make')
    print(m[1].shape[0])
    va, ea = find_adjacents(m[1])
    t.report('find adj new')
    # va, ea = find_adjacents_old(m[1])
    # t.report('find adj old')

if __name__ == "__main__":
    test_bench_find_adjacency()
