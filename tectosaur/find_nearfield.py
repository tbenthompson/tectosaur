import numpy as np
import scipy.spatial
from tectosaur.util.timer import Timer

def remove_adjacents(near_tris, edge_adj, vert_adj):
    for adj_list in [edge_adj, vert_adj]:
        for pair in adj_list:
            near_tris[pair[0]] -= {pair[1]}
            near_tris[pair[1]] -= {pair[0]}

def build_pairs(near_tris):
    nearfield_pairs = []
    for i in range(len(near_tris)):
        for other_tri in near_tris[i]:
            nearfield_pairs.append((i, other_tri))
    return nearfield_pairs

# TODO: This could be optimized quite a lot by using a specially designed
# kdtree data structure that contains bounding spheres instead of
# points. I think this could be more correct too since there are edge
# cases at the moment that fail when a small triangle is near a large
# triangle.
def find_close_or_touching(pts, tris, threshold):
    tri_pts = pts[tris]
    tri_centroid = np.sum(tri_pts, axis = 1) / 3.0
    tri_r = np.sqrt(np.max(
        np.sum((tri_pts - tri_centroid[:,np.newaxis,:]) ** 2, axis = 2),
        axis = 1
    ))

    kd = scipy.spatial.cKDTree(tri_centroid, leafsize = 1)

    near_tris = []
    for i in range(tris.shape[0]):
        result = kd.query_ball_point(tri_centroid[i,:], threshold * tri_r[i])
        without_self = set(result) - {i}
        near_tris.append(without_self)
    return near_tris

def find_nearfield(pts, tris, vert_adj, edge_adj, threshold):
    near_tris = find_close_or_touching(pts, tris, threshold)
    remove_adjacents(near_tris, edge_adj, vert_adj)
    nearfield_pairs = build_pairs(near_tris)
    return nearfield_pairs
