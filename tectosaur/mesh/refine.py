import numpy as np

def refine(m):
    pts, tris = m
    c0 = pts[tris[:,0]]
    c1 = pts[tris[:,1]]
    c2 = pts[tris[:,2]]
    midpt01 = (c0 + c1) / 2.0
    midpt12 = (c1 + c2) / 2.0
    midpt20 = (c2 + c0) / 2.0
    new_pts = np.vstack((pts, midpt01, midpt12, midpt20))
    new_tris = []
    first_new = pts.shape[0]
    ntris = tris.shape[0]
    for i, t in enumerate(tris):
        new_tris.append((t[0], first_new + i, first_new + 2 * ntris + i))
        new_tris.append((t[1], first_new + ntris + i, first_new + i))
        new_tris.append((t[2], first_new + 2 * ntris + i, first_new + ntris + i))
        new_tris.append((first_new + i, first_new + ntris + i, first_new + 2 * ntris + i))
    new_tris = np.array(new_tris)
    return remove_duplicate_pts((new_pts, new_tris))

def selective_refine(m, should_refine):
    new_pts = m[0].tolist()
    new_tris = []
    for i in range(m[1].shape[0]):
        t = m[1][i]
        if not should_refine[i]:
            new_tris.append(t)
        else:
            first_new = len(new_pts)
            c0 = m[0][t[0]]
            c1 = m[0][t[1]]
            c2 = m[0][t[2]]
            new_pts.append((c0 + c1) / 2)
            new_pts.append((c1 + c2) / 2)
            new_pts.append((c2 + c0) / 2)
            new_tris.append((t[0], first_new, first_new + 2))
            new_tris.append((t[1], first_new + 1, first_new))
            new_tris.append((t[2], first_new + 2, first_new + 1))
            new_tris.append((first_new, first_new + 1, first_new + 2))
    new_pts = np.array(new_pts)
    new_tris = np.array(new_tris)
    return remove_duplicate_pts((new_pts, new_tris))

def refine_to_size(m, threshold, fields = None):
    if fields is None:
        fields = []

    pts, tris = m
    new_pts = pts.tolist()
    new_fields = [[] for f in fields]
    new_tris = []
    for i, t in enumerate(tris):
        t_pts = pts[t]
        t_fields = [f[i].tolist() for f in fields]

        area = geometry.tri_area(t_pts)[0]
        if area < threshold:
            new_tris.append(t)
            for i in range(len(fields)):
                new_fields[i].append(t_fields[i])
            continue

        # find the longest edge
        # split in two along that edge.
        long_edge = get_longest_edge(get_edge_lens(t_pts))

        if long_edge == 0:
            edge_indices = [0, 1]
            tri_indices = [[0, 3, 2], [1, 2, 3]]
        elif long_edge == 1:
            edge_indices = [1, 2]
            tri_indices = [[0, 1, 3], [0, 3, 2]]
        elif long_edge == 2:
            edge_indices = [2, 0]
            tri_indices = [[0, 1, 3], [2, 3, 1]]

        tri_pt_indices = t.tolist()
        tri_pt_indices.append(len(new_pts))
        for t_f in t_fields:
            t_f.append(np.sum([t_f[idx] for idx in edge_indices], axis = 0) / 2.0)

        new_pts.append(np.sum(t_pts[edge_indices], axis = 0) / 2.0)
        for k in range(2):
            new_tris.append([tri_pt_indices[idx] for idx in tri_indices[k]])
            for i in range(len(fields)):
                new_fields[i].append([t_fields[i][idx] for idx in tri_indices[k]])

    if len(new_tris) > tris.shape[0]:
        out_m = (np.array(new_pts), np.array(new_tris))
        np_new_fields = [np.array(new_f) for new_f in new_fields]
        return refine_to_size(out_m, threshold, np_new_fields)

    return m, fields
