import numpy as np

def make_strip(N, xmin = 0, xmax = 100):
    xs = np.linspace(xmin, xmax, N + 1)
    tris = []
    pts = []
    for i in range(N):
        npts = len(pts)
        tris.append([npts, npts + 1, npts + 2])
        tris.append([npts + 2, npts + 1, npts + 3])
        pts.append((xs[i], 0, 0))
        pts.append((xs[i + 1], 0, 0))
        pts.append((xs[i], 1, 0))
        pts.append((xs[i + 1], 1, 0))
    return pts, tris

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
        return i * ny + j

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
