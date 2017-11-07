import numpy as np
import tectosaur.mesh.find_near_adj as find_near_adj
import tectosaur.id_problems as id_problems

def test_check_for_slivers():
    pass

# Also check for almost intersections where the triangles are really close?
# https://stackoverflow.com/questions/7113344/find-whether-two-triangles-intersect-or-not
def test_check_for_intersections():
    pass

def test_check_for_min_edge_adj_angle():
    pass

def test_check_edge_adj_tris_near_in_size():
    pts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,-1,0]])
    tris = np.array([[0,1,2], [1,0,3]])
    close_or_touch_pairs = find_near_adj.find_close_or_touching(pts, tris, 2.0)
    nearfield_pairs, va, ea = find_near_adj.split_adjacent_close(close_or_touch_pairs, tris)
    print(ea)
