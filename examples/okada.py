import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import okada_wrapper
import scipy.spatial

import tectosaur.mesh as mesh
from tectosaur.sparse_integral_op import SparseIntegralOp, FMMIntegralOp
from tectosaur.dense_integral_op import DenseIntegralOp

from solve import iterative_solve, direct_solve
import tectosaur.constraints as constraints
from tectosaur.util.timer import Timer


def build_constraints(surface_tris, fault_tris, pts):
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    cs = constraints.continuity_constraints(surface_tris, fault_tris, pts)

    # X component = 1
    # Y comp = Z comp = 0
    cs.extend(constraints.constant_bc_constraints(
        n_surf_tris, n_surf_tris + n_fault_tris, [1, 0, 0]
    ))
    cs = sorted(cs, key = lambda x: x[0][0][1])
    return cs

def refined_free_surface():
    w = 20
    minsize = 0.05
    slope = 200
    maxsize = 25
    pts = np.array([[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]])
    tris = np.array([[0, 1, 3], [3, 1, 2]])

    addedpts = 4
    it = 0
    while addedpts > 0:
        pts, tris = mesh.remove_duplicate_pts((pts, tris))
        print(it, addedpts)
        it += 1
        newpts = pts.tolist()
        newtris = []
        addedpts = 0
        # size = np.linalg.norm(np.cross(
        #     pts[tris[:, 1]] - pts[tris[:, 0]],
        #     pts[tris[:, 2]] - pts[tris[:, 0]]
        # ), axis = 1) / 2.0
        # centroid = np.mean(pts[tris], axis = 2)
        # r2 = np.sum(centroid ** 2, axis = 1)
        for i, t in enumerate(tris):
            size = np.linalg.norm(np.cross(
                pts[t[1]] - pts[t[0]],
                pts[t[2]] - pts[t[0]]
            )) / 2.0
            centroid = np.mean(pts[t], axis = 0)
            A = (centroid[0] / 1.5) ** 2 + (centroid[1] / 1.0) ** 2
            # A = np.sum(centroid ** 2)
            if (A < slope * size and size > minsize) or size > maxsize:
            # print(r2[i], size[i])
            # if r2[i] < size[i] and size[i] > minsize:
                newidx = len(newpts)
                newpts.extend([
                    (pts[t[0]] + pts[t[1]]) / 2,
                    (pts[t[1]] + pts[t[2]]) / 2,
                    (pts[t[2]] + pts[t[0]]) / 2
                ])
                newtris.extend([
                    [t[0], newidx, newidx + 2],
                    [newidx, t[1], newidx + 1],
                    [newidx + 1, t[2], newidx + 2],
                    [newidx + 1, newidx + 2, newidx]
                ])
                addedpts += 3
            else:
                newtris.append(t)
        pts = np.array(newpts)
        tris = np.array(newtris)
    final_tris = scipy.spatial.Delaunay(np.array([pts[:,0],pts[:,1]]).T).simplices
    plt.triplot(pts[:, 0], pts[:, 1], final_tris, linewidth = 0.5)
    plt.show()
    print('npts: ' + str(pts.shape[0]))
    print('ntris: ' + str(final_tris.shape[0]))
    return pts, final_tris

def make_free_surface():
    w = 6
    corners = [[-w, -w, 0], [-w, w, 0], [w, w, 0], [w, -w, 0]]
    return mesh.rect_surface(30, 30, corners)

def make_fault(L, top_depth):
    return mesh.rect_surface(10, 10, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])

def make_meshes(fault_L, top_depth):
    # surface = make_free_surface()
    surface = refined_free_surface()
    fault = make_fault(fault_L, top_depth)
    all_mesh = mesh.concat(surface, fault)
    surface_tris = all_mesh[1][:surface[1].shape[0]]
    fault_tris = all_mesh[1][surface[1].shape[0]:]
    return all_mesh, surface_tris, fault_tris

def test_okada():
    sm = 1.0
    pr = 0.25
    fault_L = 1.0
    top_depth = -0.5
    load_soln = False

    if not load_soln:
        timer = Timer()
        all_mesh, surface_tris, fault_tris = make_meshes(fault_L, top_depth)
        timer.report("Mesh")

        cs = build_constraints(surface_tris, fault_tris, all_mesh[0])
        timer.report("Constraints")

        surface_pt_idxs = np.unique(surface_tris)
        obs_pts = all_mesh[0][surface_pt_idxs,:]

        eps = [0.08, 0.04, 0.02, 0.01]
        # iop = FMMIntegralOp(
        #     eps, 18, 13, 6, 3, 7, 3.0, sm, pr, all_mesh[0], all_mesh[1]
        # )
        iop = SparseIntegralOp(
            eps, 18, 16, 6, 3, 6, 4.0,
            'H', sm, pr, all_mesh[0], all_mesh[1],
            use_tables = True
        )
        # iop = DenseIntegralOp(
        #     eps, 18, 16, 6, 3, 6, 4.0, 'H', sm, pr, all_mesh[0], all_mesh[1]
        # )
        timer.report("Integrals")

        soln = iterative_solve(iop, cs)
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
        # with open('okada.npy', 'wb') as f:
        #     pickle.dump((vals, obs_pts, surface_tris, fault_L, top_depth, sm, pr), f)
    else:
        pass
        # with open('okada.npy', 'rb') as f:
        #     vals, obs_pts, surface_tris, fault_L, top_depth, sm, pr = pickle.load(f)

    u = okada_exact(obs_pts, fault_L, top_depth, sm, pr)
    cond1 = np.logical_and(obs_pts[:,1] > -0.4, obs_pts[:,1] < -0.25)
    cond2 = np.logical_and(obs_pts[:,1] < 0.4, obs_pts[:,1] > 0.25)
    plt.plot(obs_pts[cond1, 0], vals[cond1,0],'r.')
    plt.plot(obs_pts[cond2, 0], vals[cond2,0],'r.')
    plt.plot(obs_pts[cond1, 0], u[cond1,0],'b.')
    plt.plot(obs_pts[cond2, 0], u[cond2,0],'b.')
    plt.show()
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
    print("L2diff: " + str(l2diff))
    print("L2correct: " + str(l2correct))
    print("L2relerr: " + str(l2diff / l2correct))
    print("maxerr: " + str(np.max(np.abs(diff))))

if __name__ == '__main__':
    test_okada()
