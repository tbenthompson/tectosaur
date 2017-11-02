import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import okada_wrapper
import scipy.spatial

import tectosaur
import tectosaur.mesh.refine as mesh_refine
import tectosaur.mesh.modify as mesh_modify
import tectosaur.mesh.mesh_gen as mesh_gen
import tectosaur.constraints as constraints
from tectosaur.constraint_builders import continuity_constraints, \
    constant_bc_constraints, free_edge_constraints
from tectosaur.util.timer import Timer
from tectosaur.interior import interior_integral
from tectosaur.ops.sparse_integral_op import SparseIntegralOp, FMMFarfieldBuilder
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp

from solve import iterative_solve, direct_solve

from tectosaur.util.logging import setup_root_logger
logger = setup_root_logger(__name__)

def build_constraints(surface_tris, fault_tris, pts):
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    cs = continuity_constraints(surface_tris, fault_tris)

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
    m = mesh_gen.make_rect(n_fault, n_fault, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])
    return m

def make_meshes(fault_L, top_depth, n_surf, n_fault):
    t = Timer()
    surf_w = 10
    surface = make_free_surface(surf_w, n_surf)
    t.report('make free surface')
    fault = make_fault(fault_L, top_depth, n_fault)
    t.report('make fault')
    all_mesh = mesh_modify.concat(surface, fault)
    t.report('concat meshes')
    surface_tris = all_mesh[1][:surface[1].shape[0]]
    fault_tris = all_mesh[1][surface[1].shape[0]:]
    return all_mesh, surface_tris, fault_tris

def test_okada(n_surf, n_fault = None):
    sm = 1.0
    pr = 0.25
    k_params = [sm, pr]
    fault_L = 1.0
    top_depth = 0.0
    load_soln = False
    float_type = np.float32
    if n_fault is None:
        n_fault = 7#max(2, n_surf // 5)

    timer = Timer()
    all_mesh, surface_tris, fault_tris = make_meshes(fault_L, top_depth, n_surf, n_fault)

    mesh_gen.plot_mesh3d(*all_mesh)
    timer.report('make meshes')
    logger.info('n_elements: ' + str(all_mesh[1].shape[0]))

    if not load_soln:

        T_op = SparseIntegralOp(
            6, 2, 5, 2.0,
            'elasticT3', k_params, all_mesh[0], all_mesh[1],
            float_type,
            farfield_op_type = FMMFarfieldBuilder(150, 3.0, 450)
        )
        timer.report("Integrals")

        cs = build_constraints(surface_tris, fault_tris, all_mesh[0])
        timer.report("Constraints")

        mass_op = MassOp(3, all_mesh[0], all_mesh[1])
        iop = SumOp([T_op, mass_op])
        timer.report('mass op/sum op')

        soln = iterative_solve(iop, cs, tol = 1e-6)
        # soln = direct_solve(iop, cs)
        timer.report("Solve")

        obs_pts = all_mesh[0]

        disp = soln.reshape((-1, 3, 3))
        all_vals = np.empty((all_mesh[0].shape[0], 3))
        for b in range(3):
            for d in range(3):
                all_vals[all_mesh[1][:,b],d] = disp[:,b,d]

        vals = all_vals[:,:]
        timer.report("Extract surface displacement")
        with open('okada.npy', 'wb') as f:
            pickle.dump((soln, vals, obs_pts, surface_tris, fault_L, top_depth, sm, pr), f)
    else:
        with open('okada.npy', 'rb') as f:
            soln, vals, obs_pts, surface_tris, fault_L, top_depth, sm, pr = pickle.load(f)

    u = okada_exact(obs_pts, fault_L, top_depth, sm, pr)
    # plot_results(obs_pts, surface_tris, u, vals)
    results_xsec(all_mesh[0], surface_tris, soln)
    # plot_interior_displacement(k_params, all_mesh, soln, float_type)
    # plot_interior_displacement2(k_params, all_mesh, soln, float_type)
    return print_error(obs_pts, u, vals)

def results_xsec(pts, surface_tris, soln):
    xsec_pts = []
    xsec_idxs = []
    xsec_vals = []
    for i in range(surface_tris.shape[0]):
        for pt_idx in range(3):
            p = pts[surface_tris[i,pt_idx],:]
            if np.abs(p[0]) > 0.001:
                continue
            xsec_pts.append(p)
            xsec_vals.append([soln[i * 9 + pt_idx * 3 + d] for d in range(3)])
            xsec_idxs.append([i * 9 + pt_idx * 3 + d for d in range(3)])
    xsec_pts = np.array(xsec_pts)
    xsec_vals = np.array(xsec_vals)
    print(xsec_pts)
    plt.plot(xsec_pts[:,1], xsec_vals[:,0], 'o-')
    plt.show()

def plot_interior_displacement2(k_params, all_mesh, soln, float_type):
    nxy = 300
    nz = 300
    d = 0
    xs = np.linspace(-10, 10, nxy)
    zs = np.linspace(-0.1, -4.0, nz)
    X, Y, Z = np.meshgrid(xs, xs, zs)
    obs_pts = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T.copy()
    t = Timer()
    interior_disp = -interior_integral(
        obs_pts, obs_pts, all_mesh, soln, 'elasticT3', 3, 8, k_params, float_type,
        fmm_params = [100, 3.0, 3000, 25]
    ).reshape((nxy, nxy, nz, 3))
    t.report('eval %.2E interior pts' % obs_pts.shape[0])
    # for i in range(nz):
    #     plt.figure()
    #     plt.pcolor(xs, xs, interior_disp[:,:,i,d])
    #     plt.colorbar()
    #     plt.title('at z = ' + ('%.3f' % zs[i]) + '    u' + ['x', 'y', 'z'][d])
    #     plt.show()

# def plot_interior_displacement(k_params, all_mesh, soln, float_type):
#     xs = np.linspace(-10, 10, 100)
#     for i, z in enumerate(np.linspace(0.1, 4.0, 10)):
#     # for i, z in [(0, 1.0)]:
#         X, Y = np.meshgrid(xs, xs)
#         obs_pts = np.array([X.flatten(), Y.flatten(), -z * np.ones(X.size)]).T.copy()
#         # exact_disp = okada_exact(obs_pts, fault_L, top_depth, sm, pr)
#         interior_disp = -interior_integral(
#             obs_pts, obs_pts, all_mesh, soln, 'elasticT3', 3, 8, k_params, float_type
#         ).reshape((-1, 3))
#         # for d in range(1):
#         #     plt.figure()
#         #     plt.imshow(exact_disp[:,d].reshape((xs.shape[0], -1)), interpolation = 'none')
#         #     plt.colorbar()
#         #     plt.title('exact u' + ['x', 'y', 'z'][d])
#         d = 0
#         plt.figure()
#         plt.pcolor(
#             xs, xs,
#             interior_disp[:,d].reshape((xs.shape[0], -1)),
#         )
#         # plt.colorbar()
#         plt.title('at z = ' + ('%.3f' % z) + '    u' + ['x', 'y', 'z'][d])
#         plt.show()
#         # plt.savefig('okada_depth_animation/' + str(i) + '.png')

def okada_exact(obs_pts, fault_L, top_depth, sm, pr):
    lam = 2 * sm * pr / (1 - 2 * pr)
    alpha = (lam + sm) / (lam + 2 * sm)
    print(lam, sm, pr, alpha)

    n_pts = obs_pts.shape[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = obs_pts[i, :]
        [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
            alpha, pt, -top_depth, 90.0,
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
    not0 = np.abs(pts[:,1]) > 1e-5
    test = np.logical_and(close, not0)
    diff = correct[test,:] - est[test,:]
    l2diff = np.sum(diff ** 2)
    l2correct = np.sum(correct[test,:] ** 2)
    linferr = np.max(np.abs(diff))
    print("L2diff: " + str(l2diff))
    print("L2correct: " + str(l2correct))
    print("L2relerr: " + str(l2diff / l2correct))
    print("maxerr: " + str(linferr))
    return linferr

if __name__ == '__main__':
    t = Timer()
    if len(sys.argv) == 3:
        test_okada(int(sys.argv[1]), n_fault = int(sys.argv[2]))
    else:
        test_okada(int(sys.argv[1]))
    t.report('okada')

    # import logging
    # tectosaur.logger.setLevel(logging.DEBUG)

    #n = [8, 16, 32, 64, 128, 256]
    #l2 = [0.0149648012534, 0.030572079265, 0.00867837671259, 0.00105034618493, 6.66984415273e-05, 4.07689295549e-06]
    #linf = [0.008971091166208367, 0.014749192806577716, 0.0093510756645549115, 0.0042803891552975898, 0.0013886177492512669, 0.000338113427521]
