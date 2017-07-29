import numpy as np

from cppimport import cppimport
fast_adjacency = cppimport('tectosaur.mesh.fast_adjacency')

find_touching_pts = fast_adjacency.find_touching_pts
find_adjacents = fast_adjacency.find_adjacents

def find_free_edges(tris):
    edges = dict()
    for i, t in enumerate(tris):
        for d in range(3):
            pt1_idx = t[d]
            pt2_idx = t[(d + 1) % 3]
            if pt1_idx > pt2_idx:
                pt2_idx,pt1_idx = pt1_idx,pt2_idx
            pt_pair = (pt1_idx, pt2_idx)
            edges[pt_pair] = edges.get(pt_pair, []) + [(i, d)]

    free_edges = []
    for k,e in edges.items():
        if len(e) > 1:
            continue
        free_edges.append(e[0])

    return free_edges


def rotate_tri(clicks):
    return [np.mod(clicks, 3), np.mod((1 + clicks), 3), np.mod((2 + clicks), 3)]

#TODO: Prep functions should be moved closer to nearfield_op
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

def edge_adj_orient(touching_verts):
    tv = sorted(touching_verts)
    if tv[0] == 0:
        if tv[1] == 2:
            return 2
        return 0
    return 1

def edge_adj_clicks(pair):
    obs_clicks = edge_adj_orient([pair[2][0][0], pair[2][1][0]])
    src_clicks = edge_adj_orient([pair[2][0][1], pair[2][1][1]])
    return obs_clicks, src_clicks

def edge_adj_prep(tris, ea):
    out = adj_prep(tris, ea, edge_adj_clicks)
    # These assertions are not true with triple junctions...
    # assert(np.all(out[3][:, 0] == out[4][:, 1]))
    # assert(np.all(out[3][:, 1] == out[4][:, 0]))
    return out
