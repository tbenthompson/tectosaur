import numpy as np

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

def mesh_concat(m1, m2):
    newm = np.vstack((m1[0], m2[0])), np.vstack((m1[1], m2[1] + m1[0].shape[0]))
    return remove_duplicate_pts(newm)

# Corners are ordered: lower left, lower right, upper right, upper left
def rect_surface_points(nx, ny, corners):
    corners = np.array(corners)

    rect_basis = [
        lambda x, y: x * y,
        lambda x, y: (1 - x) * y,
        lambda x, y: (1 - x) * (1 - y),
        lambda x, y: x * (1 - y)
    ]

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    vertices = np.vstack((X.reshape(nx * ny), Y.reshape(nx * ny))).T

    pts = np.sum([
        np.outer(rect_basis[i](vertices[:,0], vertices[:,1]), corners[i, :])
        for i in range(4)
    ], axis = 0)
    return pts

def rect_surface_topology(nx, ny):
    def v_idx(i, j):
        return j * nx + i

    tris = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            top_left = v_idx(i, j)
            top_right = v_idx(i + 1, j)
            bottom_left = v_idx(i, j + 1)
            bottom_right = v_idx(i + 1, j + 1)
            tris.append([top_left, bottom_left, top_right])
            tris.append([bottom_left, bottom_right, top_right])
    return np.array(tris, dtype = np.int)

def rect_surface(nx, ny, corners):
    return rect_surface_points(nx, ny, corners), rect_surface_topology(nx, ny)
