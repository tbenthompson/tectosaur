import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import okada_wrapper
import scipy.spatial

import tectosaur.mesh as mesh
from tectosaur.mass_op import MassOp
from tectosaur.sparse_integral_op import SparseIntegralOp, FMMIntegralOp
from tectosaur.dense_integral_op import DenseIntegralOp
import tectosaur.constraints as constraints

from solve import iterative_solve, direct_solve, SumOp
from tectosaur.util.timer import Timer

def spherify(center, r, pts):
    D = scipy.spatial.distance.cdist(pts, center.reshape((1,center.shape[0])))
    return (r / D) * (pts - center) + center

def make_sphere(center, r, refinements):
    pts = np.array([[0,-r,0],[r,0,0],[0,0,r],[-r,0,0],[0,0,-r],[0,r,0]])
    pts += center
    tris = np.array([[1,0,2],[2,0,3],[3,0,4],[4,0,1],[5,1,2],[5,2,3],[5,3,4],[5,4,1]])
    m = pts, tris
    for i in range(refinements):
        m = mesh.refine(m)
    spherified_m = [spherify(center, r, m[0]), m[1]]
    return spherified_m

def plot_sphere_3d(pts, tris):
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = Axes3D(fig)
    verts = pts[tris]
    coll = Poly3DCollection(verts)
    coll.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.add_collection3d(coll)
    plt.show()

def main():
    refine = 4
    m = make_sphere(np.array([0,0,0]), 1.0, refine)
    print(m[1].shape)
    # plot_sphere_3d(*m)

    sm = 1.0
    pr = 0.25
    cs = constraints.constraints(m[1], np.array([]), m[0])
    eps = [0.08, 0.04, 0.02, 0.01]
    t = Timer()
    Uop = DenseIntegralOp(
        eps, (17,17,17,17), (23,13,9,15), 7, 3,
        'U', 1.0, 0.25, m[0], m[1]
    )
    t.report('U')
    Top = DenseIntegralOp(
        eps, (14,14,14,14), (19,12,8,14), 7, 3,
        'T', 1.0, 0.25, m[0], m[1]
    )
    t.report('T')
    selfop = MassOp(2, m[0], m[1])

    # solving: u(x) + int(T*u) = int(U*t)
    # traction values radial direction because the sphere is centered at (0,0,0)
    tri_pts = m[0][m[1]]
    traction = tri_pts.reshape((m[1].shape[0] * 9))

    rhs = Uop.dot(traction)

    # full_op = SumOp(selfop, Top)
    # soln = iterative_solve(full_op, rhs, cs)

    full_op = Top.mat + selfop.mat
    cm, c_rhs = constraints.build_constraint_matrix(cs, full_op.shape[0])

    cm = cm.tocsr().todense()
    cmT = cm.T

    constrained_op = cmT.dot(full_op.dot(cm))
    rhs_constrained = cmT.dot(-full_op.dot(rhs + c_rhs).T)
    constrained_soln = np.linalg.solve(constrained_op, rhs_constrained)
    soln = cm.dot(constrained_soln)

    import ipdb; ipdb.set_trace()
    disp = np.array(soln[:full_op.shape[0]]).reshape((int(full_op.shape[0] / 9), 3, 3))









#         surface_pt_idxs = np.unique(surface_tris)
#         obs_pts = all_mesh[0][surface_pt_idxs,:]
#
#         eps = [0.08, 0.04, 0.02, 0.01]
#         # iop = FMMIntegralOp(
#         #     eps, 18, 13, 6, 3, 7, 3.0, sm, pr, all_mesh[0], all_mesh[1]
#         # )
#         iop = SparseIntegralOp(
#             eps, (17,17,17,14), (23,13,9,15), 7, 4, 10, 5.0, sm, pr, all_mesh[0], all_mesh[1]
#         )
#         # iop = DenseIntegralOp(
#         #     eps, 18, 13, 6, 3, sm, pr, all_mesh[0], all_mesh[1]
#         # )
#         timer.report("Integrals")
#
#         soln = iterative_solve(iop, cs)
#         # soln = direct_solve(iop, cs)
#         timer.report("Solve")
#
#         disp = soln[:iop.shape[0]].reshape(
#             (int(iop.shape[0] / 9), 3, 3)
#         )[:-fault_tris.shape[0]]
#         vals = [None] * surface_pt_idxs.shape[0]
#         for i in range(surface_tris.shape[0]):
#             for b in range(3):
#                 idx = surface_tris[i, b]
#                 # if vals[idx] is not None:
#                 #     np.testing.assert_almost_equal(vals[idx], disp[i,b,:], 9)
#                 vals[idx] = disp[i,b,:]
#         vals = np.array(vals)
#         timer.report("Extract surface displacement")
#         with open('okada.npy', 'wb') as f:
#             pickle.dump((vals, obs_pts, surface_tris, fault_L, top_depth, sm, pr), f)
#     else:
#         with open('okada.npy', 'rb') as f:
#             vals, obs_pts, surface_tris, fault_L, top_depth, sm, pr = pickle.load(f)
#
#     u = okada_exact(obs_pts, fault_L, top_depth, sm, pr)
#     cond1 = np.logical_and(obs_pts[:,1] > -0.4, obs_pts[:,1] < -0.25)
#     cond2 = np.logical_and(obs_pts[:,1] < 0.4, obs_pts[:,1] > 0.25)
#     plt.plot(obs_pts[cond1, 0], vals[cond1,0],'r.')
#     plt.plot(obs_pts[cond2, 0], vals[cond2,0],'r.')
#     plt.plot(obs_pts[cond1, 0], u[cond1,0],'b.')
#     plt.plot(obs_pts[cond2, 0], u[cond2,0],'b.')
#     plt.show()
#     plot_results(obs_pts, surface_tris, u, vals)
#     print_error(obs_pts, u, vals)
#
# def okada_exact(obs_pts, fault_L, top_depth, sm, pr):
#     lam = 2 * sm * pr / (1 - 2 * pr)
#     alpha = (lam + sm) / (lam + 2 * sm)
#     print(lam, sm, pr, alpha)
#
#     n_pts = obs_pts.shape[0]
#     u = np.empty((n_pts, 3))
#     for i in range(n_pts):
#         pt = obs_pts[i, :]
#         [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
#             alpha, pt, 0.5, 90.0,
#             [-fault_L, fault_L], [-1.0, 0.0], [1.0, 0.0, 0.0]
#         )
#         if suc != 0:
#             u[i, :] = 0
#         else:
#             u[i, :] = uv
#     return u
#
# def plot_results(pts, tris, correct, est):
#     vmax = np.max(correct)
#     for d in range(3):
#         plt.figure()
#         plt.tripcolor(
#             pts[:,0], pts[:, 1], tris,
#             est[:,d], #shading='gouraud',
#             cmap = 'PuOr', vmin = -vmax, vmax = vmax
#         )
#         plt.title("u " + ['x', 'y', 'z'][d])
#         plt.colorbar()
#
#     for d in range(3):
#         plt.figure()
#         plt.tripcolor(
#             pts[:, 0], pts[:, 1], tris,
#             correct[:, d], #shading='gouraud',
#             cmap = 'PuOr', vmin = -vmax, vmax = vmax
#         )
#         plt.title("Okada u " + ['x', 'y', 'z'][d])
#         plt.colorbar()
#
#     for d in range(3):
#         plt.figure()
#         plt.tripcolor(
#             pts[:, 0], pts[:, 1], tris,
#             correct[:, d] - est[:,d], #shading='gouraud',
#             cmap = 'PuOr'
#         )
#         plt.title("Diff u " + ['x', 'y', 'z'][d])
#         plt.colorbar()
#
#     plt.show()
#
# def print_error(pts, correct, est):
#     close = np.sqrt(np.sum(pts ** 2, axis = 1)) < 4.0
#     diff = correct[close,:] - est[close,:]
#     l2diff = np.sum(diff ** 2)
#     l2correct = np.sum(correct[close,:] ** 2)
#     print("L2diff: " + str(l2diff))
#     print("L2correct: " + str(l2correct))
#     print("L2relerr: " + str(l2diff / l2correct))
#     print("maxerr: " + str(np.max(np.abs(diff))))

if __name__ == '__main__':
    main()
