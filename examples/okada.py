import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import okada_wrapper
import scipy.spatial

import tectosaur
import tectosaur.mesh.modify as mesh_modify
import tectosaur.mesh.mesh_gen as mesh_gen
import tectosaur.constraints as constraints
from tectosaur.constraint_builders import continuity_constraints, \
    constant_bc_constraints, free_edge_constraints
from tectosaur.util.timer import Timer
from tectosaur.interior import interior_integral
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FMMFarfield
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp

from solve import iterative_solve, direct_solve

logger = tectosaur.setup_logger(__name__)

def build_constraints(surface_tris, fault_tris, pts):
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    cs = continuity_constraints(surface_tris, fault_tris, pts)

    # X component = 1
    # Y comp = Z comp = 0
    cs.extend(constant_bc_constraints(
        n_surf_tris, n_surf_tris + n_fault_tris, [1.0, 0.0, 0.0]
    ))
    cs.extend(free_edge_constraints(surface_tris))

    return cs

def make_free_surface(w, n):
    corners = [[-w, -w, 0], [-w, w, 0], [w, w, 0], [w, -w, 0]]
    return mesh_gen.make_rect(n, n, corners)

def make_fault(L, top_depth, n_fault):
    return mesh_gen.make_rect(n_fault, n_fault, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])

def make_meshes(fault_L, top_depth, n_surf, n_fault):
    t = Timer()
    surface = make_free_surface(10, n_surf)
    t.report('make free surface')
    fault = make_fault(fault_L, top_depth, n_fault)
    t.report('make fault')
    all_mesh = mesh_modify.concat(surface, fault)
    t.report('concat meshes')
    surface_tris = all_mesh[1][:surface[1].shape[0]]
    fault_tris = all_mesh[1][surface[1].shape[0]:]
    return all_mesh, surface_tris, fault_tris

def test_okada():
    sm = 1.0
    pr = 0.25
    k_params = [sm, pr]
    fault_L = 1.0
    top_depth = -0.5
    load_soln = False
    n_surf = 150
    n_fault = n_surf // 5

    timer = Timer()
    all_mesh, surface_tris, fault_tris = make_meshes(fault_L, top_depth, n_surf, n_fault)
    timer.report('make meshes')
    logger.info('n_elements: ' + str(all_mesh[1].shape[0]))

    # to check that the fault-surface alignment is correct
    # plt.triplot(all_mesh[0][:,0], all_mesh[0][:,1], surface_tris, linewidth = 0.3)
    # plt.triplot(all_mesh[0][:,0], all_mesh[0][:,1], fault_tris, linewidth = 0.3)
    # plt.plot(all_mesh[0][:,0], all_mesh[0][:,1], 'o', markersize = 3)
    # plt.show()

    if not load_soln:

        cs = build_constraints(surface_tris, fault_tris, all_mesh[0])
        timer.report("Constraints")

        surface_pt_idxs = np.unique(surface_tris)
        obs_pts = all_mesh[0][surface_pt_idxs,:]


        T_op = SparseIntegralOp(
            6, 2, 6, 4.0,
            'elasticT3', k_params, all_mesh[0], all_mesh[1],
            farfield_op_type = FMMFarfield
        )
        timer.report("Integrals")

        mass_op = MassOp(3, all_mesh[0], all_mesh[1])
        iop = SumOp([T_op, mass_op])
        timer.report('mass op/sum op')

        soln = iterative_solve(iop, cs, tol = 1e-6)
        # soln = direct_solve(iop, cs)
        timer.report("Solve")

        disp = soln[:iop.shape[0]].reshape(
            (int(iop.shape[0] / 9), 3, 3)
        )[:-fault_tris.shape[0]]
        vals = [None] * surface_pt_idxs.shape[0]
        for i in range(surface_tris.shape[0]):
            for b in range(3):
                idx = surface_tris[i, b]
                # if vals[idx] is not None:
                #     np.testing.assert_almost_equal(vals[idx], disp[i,b,:], 9)
                vals[idx] = disp[i,b,:]
        vals = np.array(vals)
        timer.report("Extract surface displacement")
        with open('okada.npy', 'wb') as f:
            pickle.dump((soln, vals, obs_pts, surface_tris, fault_L, top_depth, sm, pr), f)
    else:
        with open('okada.npy', 'rb') as f:
            soln, vals, obs_pts, surface_tris, fault_L, top_depth, sm, pr = pickle.load(f)

    u = okada_exact(obs_pts, fault_L, top_depth, sm, pr)
    # plot_results(obs_pts, surface_tris, u, vals)
    # plot_interior_displacement(fault_L, top_depth, k_params, all_mesh, soln)
    return print_error(obs_pts, u, vals)

def plot_interior_displacement(fault_L, top_depth, k_params, all_mesh, soln):
    xs = np.linspace(-10, 10, 100)
    # for i, z in enumerate(np.linspace(0.1, 4.0, 100)):
    for i, z in [(0, 1.0)]:
        X, Y = np.meshgrid(xs, xs)
        obs_pts = np.array([X.flatten(), Y.flatten(), -z * np.ones(X.size)]).T
        # exact_disp = okada_exact(obs_pts, fault_L, top_depth, sm, pr)
        interior_disp = -interior_integral(
            obs_pts, obs_pts, all_mesh, soln, 'elasticT3', 3, 8, k_params
        ).reshape((-1, 3))
        # for d in range(1):
        #     plt.figure()
        #     plt.imshow(exact_disp[:,d].reshape((xs.shape[0], -1)), interpolation = 'none')
        #     plt.colorbar()
        #     plt.title('exact u' + ['x', 'y', 'z'][d])
        d = 0
        plt.figure()
        plt.pcolor(
            xs, xs,
            interior_disp[:,d].reshape((xs.shape[0], -1)),
        )
        # plt.colorbar()
        plt.title('at z = ' + ('%.3f' % z) + '    u' + ['x', 'y', 'z'][d])
        plt.show()
        # plt.savefig('okada_depth_animation/' + str(i) + '.png')

def okada_exact(obs_pts, fault_L, top_depth, sm, pr):
    lam = 2 * sm * pr / (1 - 2 * pr)
    alpha = (lam + sm) / (lam + 2 * sm)
    print(lam, sm, pr, alpha)

    n_pts = obs_pts.shape[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = obs_pts[i, :]
        [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
            alpha, pt, 0.5, 90.0,
            [-fault_L, fault_L], [-1.0, 0.0], [1.0, 0.0, 0.0]
        )
        if suc != 0:
            u[i, :] = 0
        else:
            u[i, :] = uv
    return u

def plot_results(pts, tris, correct, est):
    vmax = np.max(correct)
    for d in range(3):
        plt.figure()
        plt.tripcolor(
            pts[:,0], pts[:, 1], tris,
            est[:,d], #shading='gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("u " + ['x', 'y', 'z'][d])
        plt.colorbar()

    for d in range(3):
        plt.figure()
        plt.tripcolor(
            pts[:, 0], pts[:, 1], tris,
            correct[:, d], #shading='gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("Okada u " + ['x', 'y', 'z'][d])
        plt.colorbar()

    for d in range(3):
        plt.figure()
        plt.tripcolor(
            pts[:, 0], pts[:, 1], tris,
            correct[:, d] - est[:,d], #shading='gouraud',
            cmap = 'PuOr'
        )
        plt.title("Diff u " + ['x', 'y', 'z'][d])
        plt.colorbar()

    plt.show()

def print_error(pts, correct, est):
    close = np.sqrt(np.sum(pts ** 2, axis = 1)) < 4.0
    diff = correct[close,:] - est[close,:]
    l2diff = np.sum(diff ** 2)
    l2correct = np.sum(correct[close,:] ** 2)
    linferr = np.max(np.abs(diff))
    print("L2diff: " + str(l2diff))
    print("L2correct: " + str(l2correct))
    print("L2relerr: " + str(l2diff / l2correct))
    print("maxerr: " + str(linferr))
    return linferr

if __name__ == '__main__':
    test_okada()

    # import logging
    # tectosaur.logger.setLevel(logging.DEBUG)

    #n = [8, 16, 32, 64, 128, 256]
    #l2 = [0.0149648012534, 0.030572079265, 0.00867837671259, 0.00105034618493, 6.66984415273e-05, 4.07689295549e-06]
    #linf = [0.008971091166208367, 0.014749192806577716, 0.0093510756645549115, 0.0042803891552975898, 0.0013886177492512669, 0.000338113427521]
