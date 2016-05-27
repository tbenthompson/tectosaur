import matplotlib.pyplot as plt
import numpy as np
import okada_wrapper
from slow_test import slow
from tectosaur.mesh import rect_surface


@slow
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

