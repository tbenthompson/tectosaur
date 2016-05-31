import numpy as np

def find_adjacents(tris):
    max_pt_idx = np.max(tris)
    touching_pt = [[] for i in range(max_pt_idx + 1)]
    for i, t in enumerate(tris):
        for d in range(3):
            touching_pt[t[d]].append((i, d))

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

def rotate_tri(tri, clicks):
    return [tri[clicks % 3], tri[(1 + clicks) % 3], tri[(2 + clicks) % 3]]

def vert_adj_prep(tris, va):
    tri_indices = np.empty((len(va), 2), dtype = np.int)
    obs_tris = np.zeros((len(va), 3), dtype = np.int)
    src_tris = np.zeros((len(va), 3), dtype = np.int)
    for i, pair in enumerate(va):
        assert(len(pair[2]) == 1)
        obs_tris[i, :] = rotate_tri(tris[pair[0]], pair[2][0][0])
        src_tris[i, :] = rotate_tri(tris[pair[1]], pair[2][0][1])
        tri_indices[i, 0] = pair[0]
        tri_indices[i, 1] = pair[1]
    assert(np.all(obs_tris[:, 0] == src_tris[:, 0]))
    return tri_indices, obs_tris, src_tris

def edge_adj_orient(touching_verts):
    tv = sorted(touching_verts)
    if tv[0] == 0:
        if tv[1] == 2:
            return 2
        return 0
    return 1

def edge_adj_prep(tris, ea):
    tri_indices = np.empty((len(ea), 2), dtype = np.int)
    obs_tris = np.zeros((len(ea), 3), dtype = np.int)
    src_tris = np.zeros((len(ea), 3), dtype = np.int)
    for i, pair in enumerate(ea):
        assert(len(pair[2]) == 2)

        obs_clicks = edge_adj_orient([pair[2][0][0], pair[2][1][0]])
        obs_tris[i, :] = rotate_tri(tris[pair[0]], obs_clicks)

        src_clicks = edge_adj_orient([pair[2][0][1], pair[2][1][1]])
        src_tris[i, :] = rotate_tri(tris[pair[1]], src_clicks)

        tri_indices[i, 0] = pair[0]
        tri_indices[i, 1] = pair[1]
    assert(np.all(obs_tris[:, 0] == src_tris[:, 1]))
    assert(np.all(obs_tris[:, 1] == src_tris[:, 0]))
    return tri_indices, obs_tris, src_tris
