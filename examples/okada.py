import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import okada_wrapper

import tectosaur.mesh as mesh
from tectosaur.integral_op import SelfIntegralOperator

from okada_solve import solve
from okada_constraints import constraints
from tectosaur.util.timer import Timer

def make_free_surface():
    w = 10
    corners = [[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]]
    return mesh.rect_surface(100,100,corners)
    n = 36
    inner_n = 15
    outer_n = 30
    one_side = (
        np.linspace(0, 3, inner_n)[1:].tolist() +
        (3.0 * 1.1 ** np.arange(1, outer_n)).tolist()
    )
    xs = np.array((-np.array(one_side[::-1])).tolist() + [0] + one_side)
    minxs = np.min(xs)
    maxxs = np.max(xs)
    xs = (xs - minxs) / (maxxs - minxs)

    w = maxxs
    corners = [[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]]
    pts = mesh.rect_surface_points(corners, xs, xs)
    topology = mesh.rect_surface_topology(xs.shape[0], xs.shape[0])
    return (pts, topology)

def make_fault(L, top_depth):
    return mesh.rect_surface(5, 5, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])

def test_okada():
    sm = 1.0
    pr = 0.25

    timer = Timer()
    fault_L = 1.0 / 3.0
    top_depth = -0.5
    surface = make_free_surface()
    fault = make_fault(fault_L, top_depth)
    all_mesh = mesh.concat(surface, fault)
    print(all_mesh[1].shape)
    surface_tris = all_mesh[1][:surface[1].shape[0]]
    fault_tris = all_mesh[1][surface[1].shape[0]:]
    timer.report("Mesh")

    cs = constraints(surface_tris, fault_tris, all_mesh[0])
    timer.report("Constraints")

    iop = SelfIntegralOperator(15, 15, 8, sm, pr, all_mesh[0], all_mesh[1])
    timer.report("Integrals")

    soln = solve(iop, cs)
    timer.report("Solve")

    disp = soln[:iop.shape[0]].reshape((int(iop.shape[0]/9), 3, 3))[:-6]
    vals = [None] * surface[0].shape[0]
    for i in range(surface[1].shape[0]):
        for b in range(3):
            idx = surface[1][i, b]
            vals[idx] = disp[i,b,:]
    vals = np.array(vals)
    timer.report("Extract surface displacement")

    # import sys; sys.exit()

    lam = 2 * sm * pr / (1 - 2 * pr)
    alpha = (lam + sm) / (lam + 2 * sm)

    timer.restart()
    n_pts = surface[0].shape[0]
    obs_pts = surface[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = surface[0][i, :]
        pt[2] = 0
        [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
            alpha, pt, 0, 90, [-fault_L, fault_L], [top_depth - 1, top_depth], [1, 0, 0]
        )
        u[i, :] = uv
    timer.report("Okada")

    vmax = np.max(u)

    triang = tri.Triangulation(surface[0][:,0], surface[0][:,1], surface[1])
    for d in range(3):
        plt.figure()
        plt.tripcolor(triang, 2 * vals[:,d], shading = 'gouraud', cmap = 'PuOr',
            vmin = -vmax, vmax = vmax)
        plt.title("u " + ['x', 'y', 'z'][d])
        plt.colorbar()

    # plt.figure()
    # plt.quiver(obs_pts[:, 0], obs_pts[:, 1], u[:, 0], u[:, 1])
    # plt.figure()
    # plt.streamplot(obs_pts[:, 0].reshape((n,n)), obs_pts[:, 1].reshape((n,n)), u[:, 0].reshape((n,n)), u[:, 1].reshape((n,n)))
    for d in range(3):
        plt.figure()
        plt.tripcolor(
            obs_pts[:, 0], obs_pts[:, 1], surface[1],
            u[:, d], shading='gouraud', cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("Okada u " + ['x', 'y', 'z'][d])
        plt.colorbar()
    plt.show()

if __name__ == '__main__':
    test_okada()
