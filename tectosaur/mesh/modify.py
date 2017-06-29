
def remove_duplicate_pts(m):
    threshold = (np.max(m[0]) - np.min(m[0])) * 1e-13
    idx_map = dict()
    next_idx = 0
    for i in range(m[0].shape[0]):
        dists = np.sum((m[0][:i] - m[0][i,:]) ** 2, axis = 1)
        close = dists < threshold ** 2
        if np.sum(close) > 0:
            replacement_idx = np.argmax(close)
            idx_map[i] = idx_map[replacement_idx]
        else:
            idx_map[i] = next_idx
            next_idx += 1

    n_pts_out = np.max(list(idx_map.values())) + 1
    out_pts = np.empty((n_pts_out, 3))
    for i in range(m[0].shape[0]):
        out_pts[idx_map[i],:] = m[0][i,:]

    out_tris = np.empty_like(m[1])
    for i in range(m[1].shape[0]):
        for d in range(3):
            out_tris[i,d] = idx_map[m[1][i,d]]

    return out_pts, out_tris

def concat_two(m1, m2):
    newm = np.vstack((m1[0], m2[0])), np.vstack((m1[1], m2[1] + m1[0].shape[0]))
    return remove_duplicate_pts(newm)

def concat(*ms):
    return reduce(lambda x,y: concat_two(x,y), ms)

def flip_normals(m):
    return (m[0], np.array([[m[1][i,0],m[1][i,2],m[1][i,1]] for i in range(m[1].shape[0])]))
