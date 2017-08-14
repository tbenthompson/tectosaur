import numpy as np
import scipy.spatial
from tectosaur.util.timer import Timer

import tectosaur.util.profile

from cppimport import cppimport
fast_find_nearfield = cppimport('tectosaur.mesh.fast_find_nearfield')

split_adjacent_close = fast_find_nearfield.split_adjacent_close

def find_close_or_touching(pts, tris, threshold):
    tri_pts = pts[tris]
    tri_centroid = np.sum(tri_pts, axis = 1) / 3.0
    tri_r = np.sqrt(np.max(
        np.sum((tri_pts - tri_centroid[:,np.newaxis,:]) ** 2, axis = 2),
        axis = 1
    ))

    out = fast_find_nearfield.get_nearfield(tri_centroid, tri_r, threshold, 50)
    return out

def rotate_tri(clicks):
    return [np.mod(clicks, 3), np.mod((1 + clicks), 3), np.mod((2 + clicks), 3)]

#TODO: Remove these prep functions
def adj_prep(tris, adj, clicks_fnc):
    n_pairs = adj.shape[0]
    tri_indices = np.empty((n_pairs, 2), dtype = np.int)
    obs_clicks = np.empty(n_pairs, dtype = np.int)
    src_clicks = np.empty(n_pairs, dtype = np.int)
    obs_tris = np.zeros((n_pairs, 3), dtype = np.int)
    src_tris = np.zeros((n_pairs, 3), dtype = np.int)
    for i in range(n_pairs):
        pair = (adj[i,0], adj[i,1], adj[i,2:].reshape((-1, 2)).tolist())
        obs_clicks[i], src_clicks[i] = clicks_fnc(pair)
        obs_rot = rotate_tri(obs_clicks[i])
        src_rot = rotate_tri(src_clicks[i])
        obs_tris[i, :] = tris[pair[0]][obs_rot]
        src_tris[i, :] = tris[pair[1]][src_rot]
        tri_indices[i, 0] = pair[0]
        tri_indices[i, 1] = pair[1]
    return tri_indices, obs_clicks, src_clicks, obs_tris, src_tris

def vert_adj_clicks(pair):
    assert(len(pair[2]) == 1)
    return pair[2][0][0], pair[2][0][1]

def vert_adj_prep(tris, va):
    out = adj_prep(tris, va, vert_adj_clicks)
    assert(np.all(out[3][:, 0] == out[4][:, 0]))
    return out
