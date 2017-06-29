import numpy as np

def find_touching_pts(tris):
    max_pt_idx = np.max(tris)
    out = [[] for i in range(max_pt_idx + 1)]
    for i, t in enumerate(tris):
        for d in range(3):
            out[t[d]].append((i, d))
    return out

def find_adjacents(tris):
    touching_pt = find_touching_pts(tris)
    vert_adjacents = []
    edge_adjs = []
    for i, t in enumerate(tris):
        touching_tris = []
        for d in range(3):
            for other_t in touching_pt[t[d]]:
                touching_tris.append((other_t[0], d, other_t[1]))

        already = []
        for other_t in touching_tris:
            if other_t[0] in already or other_t[0] == i:
                continue
            already.append(other_t[0])

            shared_verts = []
            for other_t2 in touching_tris:
                if other_t2[0] != other_t[0]:
                    continue
                shared_verts.append((other_t2[1], other_t2[2]))

            n_shared_verts = len(shared_verts)
            if n_shared_verts == 1:
                vert_adjacents.append((i, other_t[0], shared_verts))
            elif n_shared_verts == 2:
                edge_adjs.append((i, other_t[0], shared_verts))
            else:
                raise Exception("Duplicate triangles!")

    return vert_adjacents, edge_adjs

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

def adj_prep(tris, adj, clicks_fnc):
    tri_indices = np.empty((len(adj), 2), dtype = np.int)
    obs_clicks = np.empty(len(adj), dtype = np.int)
    src_clicks = np.empty(len(adj), dtype = np.int)
    obs_tris = np.zeros((len(adj), 3), dtype = np.int)
    src_tris = np.zeros((len(adj), 3), dtype = np.int)
    for i, pair in enumerate(adj):
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
