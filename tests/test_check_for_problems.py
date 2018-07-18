import numpy as np
import tectosaur.check_for_problems as problems
import tectosaur.mesh.find_near_adj as find_near_adj
from tectosaur.nearfield.table_params import min_angle_isoceles_height

def test_check_for_slivers():
    ts = [
        (0.5, 0.5),
        (0.5, min_angle_isoceles_height),
        (0.5, min_angle_isoceles_height - 0.001),
        (0.5, min_angle_isoceles_height + 0.001),
        (0.4, min_angle_isoceles_height)
    ]
    pts = np.array([[0,0,0], [1,0,0]] + [[a,b,0] for a,b in ts])
    tris = np.array([[0, 1, 2 + i] for i in range(len(ts))])
    bad_tris = problems.check_for_slivers((pts, tris))
    assert(np.all(bad_tris == [(2, 4)]))

# Also check for almost intersections where the triangles are really close?
def test_check_for_intersections_nearfield():
    pts = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [0.1,-0.1,0],[0.5,0.5,0.5],[0.5,0.5,-0.5],
    ])
    tris = np.array([[0,1,2],[3,4,5]])
    bad_pairs = problems.check_for_intersections((pts, tris))
    assert(np.all(bad_pairs == [(0,1)]))

def test_check_for_intersections_va():
    pts = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [0.5,0.5,0.5],[0.5,0.5,-0.5],
        [-1,0,0],[0,-1,0],
    ])
    tris = np.array([[0,1,2],[0,3,4],[0,5,6]])
    bad_pairs = problems.check_for_intersections((pts, tris))
    assert(np.all(bad_pairs == [(0,1), (1, 0)]))

def test_check_for_intersections_va_inplane():
    pts = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [0.5,0.5,0.5],[0.5,0.5,0.0],
    ])
    tris = np.array([[0,1,2],[0,3,4]])
    bad_pairs = problems.check_for_intersections((pts, tris))
    assert(np.all(bad_pairs == [(0,1), (1,0)]))

def test_check_for_intersections_ea():
    pts = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [0,-1,0],[0,1,0]
    ])
    tris = np.array([[0,1,2],[0,1,3],[0,1,4]])
    bad_pairs = problems.check_for_intersections((pts, tris))
    assert(np.all(bad_pairs == [(0,2), (2,0)]))

def test_check_for_min_edge_adj_angle():
    pts = np.array([[0,0,0],[1,0,0],[0.5,0.5,0],[0.5, 0.4, 0.1],[0.5,-0.5,0.0]])
    tris = np.array([[0,1,2],[1,0,3],[1,0,4]])
    bad_pairs = problems.check_min_adj_angle((pts, tris))
    assert(np.all(bad_pairs == [(0,1), (1,0)]))

def test_check_tris_tall_enough():
    ts = [
        (0.5, 0.5),
        (0.5, min_angle_isoceles_height),
        (0.5, min_angle_isoceles_height - 0.001),
        (0.5, min_angle_isoceles_height + 0.001),
        (0.4, min_angle_isoceles_height)
    ]
    pts = np.array([[0,0,0], [1,0,0]] + [[a,b,0] for a,b in ts])
    tris = np.array([[0, 1, 2 + i] for i in range(len(ts))])
    bad_tris = problems.check_tris_tall_enough((pts, tris))
    assert(np.all(bad_tris == [(2, 4)]))

def test_check_okada():
    m = np.load('tests/okada_mesh.npy')
    intersections, slivers, short, sharp = problems.check_for_problems(m)
    assert(len(slivers) == 0)
    assert(len(short) == 0)
    assert(intersections.shape[0] == 0)
    assert(sharp.shape[0] == 0)
