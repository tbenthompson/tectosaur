import scipy.spatial
from functools import reduce
import numpy as np

import tectosaur.geometry as geometry
from tectosaur.standardize import get_edge_lens, get_longest_edge

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

# Corners are ordered: lower left, lower right, upper right, upper left
def rect_points(corners, xhat_vals, yhat_vals):
    nx = xhat_vals.shape[0]
    ny = yhat_vals.shape[0]
    corners = np.array(corners)

    rect_basis = [
        lambda x, y: x * y,
        lambda x, y: (1 - x) * y,
        lambda x, y: (1 - x) * (1 - y),
        lambda x, y: x * (1 - y)
    ]

    X, Y = np.meshgrid(xhat_vals, yhat_vals)
    vertices = np.vstack((X.reshape(nx * ny), Y.reshape(nx * ny))).T

    pts = np.sum([
        np.outer(rect_basis[i](vertices[:,0], vertices[:,1]), corners[i, :])
        for i in range(4)
    ], axis = 0)
    return pts

def rect_topology(nx, ny):
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

def make_rect(nx, ny, corners):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    return rect_points(corners, x, y), rect_topology(nx, ny)

def spherify(center, r, pts):
    D = scipy.spatial.distance.cdist(pts, center.reshape((1,center.shape[0])))
    return (r / D) * (pts - center) + center

def make_ellipse(center, rx, ry, rz):
    pts = np.array([[0,-ry,0],[rx,0,0],[0,0,rz],[-rx,0,0],[0,0,-rz],[0,ry,0]])
    tris = np.array([[1,0,2],[2,0,3],[3,0,4],[4,0,1],[5,1,2],[5,2,3],[5,3,4],[5,4,1]])
    pts += center
    return pts, tris

def make_sphere(center, r, refinements):
    center = np.array(center)
    pts = np.array([[0,-r,0],[r,0,0],[0,0,r],[-r,0,0],[0,0,-r],[0,r,0]])
    tris = np.array([[1,0,2],[2,0,3],[3,0,4],[4,0,1],[5,1,2],[5,2,3],[5,3,4],[5,4,1]])
    pts += center
    m = pts, tris
    for i in range(refinements):
        m = refine(m)
        m = (spherify(center, r, m[0]), m[1])
    m = (m[0], m[1])
    return m

def plot_mesh3d(pts, tris):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure()
    ax = Axes3D(fig)
    verts = pts[tris]
    coll = Poly3DCollection(verts)
    coll.set_facecolor((0.0, 0.0, 0.0, 0.0))
    coll.set_edgecolor((0.0, 0.0, 0.0, 1.0))
    ax.add_collection3d(coll)
    plt.show()
