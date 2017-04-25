import pickle
import numpy as np
import scipy.spatial

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.pyplot as plt

import okada_wrapper

import tectosaur.mesh as mesh
from tectosaur.mass_op import MassOp
from tectosaur.sparse_integral_op import SparseIntegralOp
from tectosaur.dense_integral_op import DenseIntegralOp
from tectosaur.quadrature import gauss2d_tri
import tectosaur.constraints as constraints
import tectosaur.geometry as geometry

from solve import iterative_solve, direct_solve
from tectosaur.util.timer import Timer

def build_constraints(inner_tris, outer_tris, full_mesh):
    cs = constraints.continuity_constraints(full_mesh[1], np.array([]), full_mesh[0])
    return constraints.sort_by_constrained_dof(cs)

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
        m = (spherify(center, r, m[0]), m[1])
    return m

# Solution from http://solidmechanics.org/text/Chapter4_1/Chapter4_1.htm
# Section 4.1.4
def correct_displacement(a, b, pa, pb, sm, pr, R):
    E = 2 * sm * (1 + pr)
    factor = 1.0 / (2 * E * (b ** 3 - a ** 3) * R ** 2)
    term1 = 2 * (pa * a ** 3 - pb * b ** 3) * (1 - 2 * pr) * R ** 3
    term2 = (pa - pb) * (1 + pr) * b ** 3 * a ** 3
    return factor * (term1 + term2)

def check_normals(tri_pts, ns):
    center = np.mean(tri_pts, axis = 1)
    plt.plot(center[:, 0], center[:, 1], '*')
    for i in range(ns.shape[0]):
        start = center[i]
        end = start + ns[i] * 0.1
        data = np.array([start, end])
        plt.plot(data[:, 0], data[:, 1], 'k')
    plt.show()

def runner(param, do_solve = True):
    refine = 2
    a = 1.0
    b = 2.0
    sm = 1.0
    pr = 0.25
    pa = 1.0
    pb = -1.0

    print(correct_displacement(a, b, pa, pb, sm, pr, np.array([a, b])))
    print(correct_displacement(a, b, -pa, pb, sm, pr, np.array([a, b])))
    print(correct_displacement(a, b, -pa, -pb, sm, pr, np.array([a, b])))
    print(correct_displacement(a, b, pa, -pb, sm, pr, np.array([a, b])))

    m_inner = make_sphere(np.array([0,0,0]), a, refine)
    m_outer = make_sphere(np.array([0,0,0]), b, refine)
    # if param:
    #     m_inner = mesh.flip_normals(m_inner)

    m = mesh.concat(m_inner, m_outer)
    print(m[1].shape)
    n_inner = m_inner[1].shape[0]
    inner_tris = m[1][:n_inner]
    outer_tris = m[1][n_inner:]
    tri_pts = m[0][m[1]]
    # mesh.plot_mesh3d(*m)

    cs = build_constraints(inner_tris, outer_tris, m)

    # solving: u(x) + int(T*u) = int(U*t)
    # values in radial direction because the sphere is centered at (0,0,0)

    unscaled_ns = geometry.unscaled_normals(tri_pts)
    ns = unscaled_ns / geometry.jacobians(unscaled_ns)[:,np.newaxis]
    # check_normals(tri_pts, ns)

    input_nd = np.tile(ns[:,np.newaxis,:], (1, 3, 1))
    input = input_nd.reshape(tri_pts.shape[0] * 9)

    # Inner surface has traction pa
    input[:n_inner] *= pa

    # Outer surface has traction pb
    input[n_inner:] *= pb

    selfop = MassOp(3, m[0], m[1]).mat

    eps = []
    t = Timer()
    Uop = DenseIntegralOp(
        eps, 20, 15, 6, 3, 6, 4.0,
        'U', sm, pr, m[0], m[1], use_tables = True, remove_sing = False
    )
    t.report('U')
    Top = DenseIntegralOp(
        eps, 20, 15, 6, 3, 6, 4.0,
        'T', sm, pr, m[0], m[1], use_tables = True, remove_sing = False
    )
    t.report('T')

    if not do_solve:
        return Top.mat
    lhs = Top.mat + selfop
    rhs = -Uop.dot(input)
    t.report('setup system')

    cm, c_rhs = constraints.build_constraint_matrix(cs, lhs.shape[0])
    t.report('build constraint matrix')
    cm = cm.tocsr().todense()
    t.report('constraints to dense')
    cmT = cm.T
    t.report('transpose constraint matrix')
    lhs_constrained = cmT.dot(lhs.dot(cm))
    rhs_constrained = cmT.dot((rhs + lhs.dot(c_rhs)).T)
    t.report('constrain')
    constrained_soln = np.linalg.solve(lhs_constrained, rhs_constrained)
    t.report('solve')
    soln = cm.dot(constrained_soln)
    t.report('deconstrain')

    disp = np.array(soln).reshape((int(lhs.shape[0] / 9), 3, 3))
    avg_face_disp = np.mean(disp, axis = 1)
    disp_mag = np.sqrt(np.sum(avg_face_disp ** 2, axis = 1))
    inner_disp_mag = disp_mag[:n_inner]
    outer_disp_mag = disp_mag[n_inner:]
    print(inner_disp_mag[:10])
    print(outer_disp_mag[:10])
    return Top.mat
    # to_plot = disp_mag

    # solve_for = 'disp'
    # if solve_for == 'disp':
    # elif solve_for == 'trac':
    #     lhs = Uop.mat
    #     rhs = (-Top.mat - selfop).dot(input)
    #     soln = np.linalg.solve(lhs, rhs.T)
    #     trac = np.array(soln).reshape((int(lhs.shape[0] / 9), 3, 3))
    #     avg_face_trac = np.mean(trac, axis = 1)
    #     trac_mag = np.sqrt(np.sum(avg_face_trac ** 2, axis = 1))
    #     to_plot = trac_mag


    # for var in [to_plot, input_mag]:
    #     fig = plt.figure()
    #     ax = fig.gca(projection='3d')
    #     cmap = plt.get_cmap('Blues')
    #     triang = tri.Triangulation(m[0][:,0], m[0][:,1], m[1])
    #     collec = ax.plot_trisurf(triang, m[0][:,2], cmap=cmap, shade=False, linewidth=0.)
    #     collec.set_array(var)
    #     collec.autoscale()
    #     plt.colorbar(collec)
    # plt.show()

def main():
    params = [3]
    outputs = [runner(p, do_solve = True) for p in params]
    for i, o in enumerate(outputs):
        np.save('debug/' + str(i) + '.npy', o)
    outputs = [np.load('debug/' + str(i) + '.npy') for i in range(len(params))]
    # logdiff = np.log10(np.abs(outputs[0] - outputs[1]))
    # logdiff = np.where(np.isinf(logdiff), np.min(logdiff), logdiff)
    # plt.imshow(logdiff, vmin = -6, vmax = -1, interpolation = 'none')
    # plt.colorbar()
    # plt.show()








if __name__ == '__main__':
    main()
