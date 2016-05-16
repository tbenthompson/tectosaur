import matplotlib.pyplot as plt
import numpy as np
import okada_wrapper

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

def test_okada():
    sm = 30e9
    pr = 0.25
    lam = 2 * sm * pr / (1 - 2 * pr)

    n = 25
    surface = rect_surface(n, n, [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]])
    fault = rect_surface(1, 1, [[-0.5, 0, 0], [-0.5, 0, -1], [0.5, 0, -1], [0.5, 0, 0]])

    alpha = (lam + sm) / (lam + 2 * sm)

    n_pts = surface[0].shape[0]
    obs_pts = surface[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = surface[0][i, :]
        [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
            alpha, pt, 0, 90, [-0.5, 0.5], [-1, 0], [1, 0, 0]
        )
        u[i, :] = uv



    # plt.figure()
    # plt.quiver(obs_pts[:, 0], obs_pts[:, 1], u[:, 0], u[:, 1])
    # plt.figure()
    # plt.streamplot(obs_pts[:, 0].reshape((n,n)), obs_pts[:, 1].reshape((n,n)), u[:, 0].reshape((n,n)), u[:, 1].reshape((n,n)))
    # for d in range(3):
    #     plt.figure()
    #     plt.tripcolor(
    #         obs_pts[:, 0], obs_pts[:, 1], surface[1],
    #         u[:, d], shading='gouraud'
    #     )
    #     plt.colorbar()
    # plt.show()

