import numpy as np
import scipy.sparse.csgraph
import scipy.sparse
import matplotlib.pyplot as plt


def tri_connectivity_graph(tris):
    n_tris = tris.shape[0]
    touching = [[] for i in range(np.max(tris) + 1)]
    for i in range(n_tris):
        for d in range(3):
            touching[tris[i, d]].append(i)

    rows = []
    cols = []
    for i in range(len(touching)):
        for row in touching[i]:
            for col in touching[i]:
                rows.append(row)
                cols.append(col)
    rows = np.array(rows)
    cols = np.array(cols)
    connectivity = scipy.sparse.coo_matrix(
        (np.ones(rows.shape[0]), (rows, cols)), shape=(n_tris, n_tris)
    )
    return connectivity


def get_connected_components(tris):
    return scipy.sparse.csgraph.connected_components(tri_connectivity_graph(tris))


def find_free_edges(tris):
    edges = dict()
    for i, t in enumerate(tris):
        for d in range(3):
            pt1_idx = t[d]
            pt2_idx = t[(d + 1) % 3]
            if pt1_idx > pt2_idx:
                pt2_idx, pt1_idx = pt1_idx, pt2_idx
            pt_pair = (pt1_idx, pt2_idx)
            edges[pt_pair] = edges.get(pt_pair, []) + [(i, d)]

    free_edges = []
    for k, e in edges.items():
        if len(e) > 1:
            continue
        free_edges.append(e[0])

    return free_edges


def get_boundary_loop(tris):
    which_comp = get_connected_components(tris)[1]
    n_surfaces = np.unique(which_comp).shape[0]
    orderings = []
    for surf_idx in range(n_surfaces):
        tri_subset = tris[which_comp == surf_idx]
        free_edges = find_free_edges(tri_subset)
        pt_to_pt = [
            (tri_subset[tri_idx, edge_idx], tri_subset[tri_idx, (edge_idx + 1) % 3])
            for tri_idx, edge_idx in free_edges
        ]

        pts_to_edges = dict()
        for i, e in enumerate(pt_to_pt):
            for lr in [0, 1]:
                pts_to_edges[e[lr]] = pts_to_edges.get(e[lr], []) + [i]

        for k, v in pts_to_edges.items():
            assert len(v) == 2

        ordering = [pt_to_pt[0][0], pt_to_pt[0][1]]
        looped = False
        while not looped:
            pt_idx = ordering[-1]
            prev_pt_idx = ordering[-2]
            for e_idx in pts_to_edges[pt_idx]:
                edge = pt_to_pt[e_idx]
                if edge[0] == prev_pt_idx or edge[1] == prev_pt_idx:
                    continue
                if edge[0] == pt_idx:
                    ordering.append(edge[1])
                else:
                    ordering.append(edge[0])
            if ordering[-1] == ordering[0]:
                looped = True
        orderings.append(ordering)
    return orderings


# pts, tris, t, slip, state = np.load('data_for_brendan.npy')
# loop = get_boundary_loop(tris)[0]

# for i in range(len(loop) - 1):
#     P = pts[[loop[i], loop[i + 1]]]
#     plt.plot(P[:,0], P[:,1])
# plt.show()
