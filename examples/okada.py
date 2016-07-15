import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import okada_wrapper

import tectosaur.mesh as mesh
from tectosaur.sparse_integral_op import SparseIntegralOperator
from tectosaur.dense_integral_op import DenseIntegralOperator

from okada_solve import iterative_solve
from okada_constraints import constraints, insert_constraints
from tectosaur.util.timer import Timer

def refined_free_surface():
    n = 36
    inner_n = 15
    outer_n = 20
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

def make_free_surface():
    w = 3
    corners = [[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]]
    return mesh.rect_surface(37,37,corners)

def make_fault(L, top_depth):
    return mesh.rect_surface(20, 20, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])

def make_meshes(fault_L, top_depth):
    surface = make_free_surface()
    fault = make_fault(fault_L, top_depth)
    all_mesh = mesh.concat(surface, fault)
    surface_tris = all_mesh[1][:surface[1].shape[0]]
    fault_tris = all_mesh[1][surface[1].shape[0]:]
    return all_mesh, surface_tris, fault_tris

def test_okada():
    sm = 1.0
    pr = 0.25
    fault_L = 0.3333
    top_depth = -0.5

    timer = Timer()
    all_mesh, surface_tris, fault_tris = make_meshes(fault_L, top_depth)
    timer.report("Mesh")

    cs = constraints(surface_tris, fault_tris, all_mesh[0])
    timer.report("Constraints")

    surface_pt_idxs = np.unique(surface_tris)
    obs_pts = all_mesh[0][surface_pt_idxs,:]
    load_soln = False
    if not load_soln:
        iop = SparseIntegralOperator(16, 16, 8, 2, 7, 4.0, sm, pr, all_mesh[0], all_mesh[1])
        # iop = DenseIntegralOperator(16, 16, 8, 4, sm, pr, all_mesh[0], all_mesh[1])
        timer.report("Integrals")

        soln = iterative_solve(iop, cs)
        timer.report("Solve")

        disp = soln[:iop.shape[0]].reshape((int(iop.shape[0] / 9), 3, 3))[:-fault_tris.shape[0]]
        vals = [None] * surface_pt_idxs.shape[0]
        for i in range(surface_tris.shape[0]):
            for b in range(3):
                idx = surface_tris[i, b]
                # if vals[idx] is not None:
                #     np.testing.assert_almost_equal(vals[idx], disp[i,b,:], 9)
                vals[idx] = disp[i,b,:]
        vals = np.array(vals)
        timer.report("Extract surface displacement")
        np.save('okada.npy', vals)
    else:
        vals = np.load('okada.npy')

    timer.restart()
    u = okada_exact(obs_pts, fault_L, top_depth, sm, pr)
    timer.report("Okada")

    plot_results(obs_pts, surface_tris, u, vals)
    print_error(obs_pts, u, vals)

def okada_exact(obs_pts, fault_L, top_depth, sm, pr):
    lam = 2 * sm * pr / (1 - 2 * pr)
    alpha = (lam + sm) / (lam + 2 * sm)
    print(lam, sm, pr, alpha)

    n_pts = obs_pts.shape[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = obs_pts[i, :]
        [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
            alpha, pt, -top_depth + 0.5, 90.0,
            [-fault_L, fault_L], [-0.5, 0.5], [1.0 / np.sqrt(3), 0.0, 0.0]
        )
        assert(suc == 0)
        u[i, :] = uv
    return u

def plot_results(pts, tris, correct, est):
    vmax = np.max(correct)
    for d in range(3):
        plt.figure()
        plt.tripcolor(
            pts[:,0], pts[:, 1], tris,
            est[:,d], shading = 'gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("u " + ['x', 'y', 'z'][d])
        plt.colorbar()

    for d in range(3):
        plt.figure()
        plt.tripcolor(
            pts[:, 0], pts[:, 1], tris,
            correct[:, d], shading='gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("Okada u " + ['x', 'y', 'z'][d])
        plt.colorbar()

    plt.show()

def print_error(pts, correct, est):
    import ipdb; ipdb.set_trace()
    close = np.sqrt(np.sum(pts ** 2, axis = 1)) < 4.0
    diff = correct[close,:] - est[close,:]
    l2diff = np.sum(diff ** 2)
    l2correct = np.sum(correct[close,:] ** 2)
    print("L2diff: " + str(l2diff))
    print("L2correct: " + str(l2correct))
    print("L2relerr: " + str(l2diff / l2correct))
    print("maxerr: " + str(np.max(np.abs(diff))))

if __name__ == '__main__':
    test_okada()
