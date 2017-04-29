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

# Solution from http://solidmechanics.org/text/Chapter4_1/Chapter4_1.htm
# Section 4.1.4
def correct_displacement(a, b, pa, pb, sm, pr, R):
    E = 2 * sm * (1 + pr)
    factor = 1.0 / (2 * E * (b ** 3 - a ** 3) * R ** 2)
    term1 = 2 * (pa * a ** 3 - pb * b ** 3) * (1 - 2 * pr) * R ** 3
    term2 = (pa - pb) * (1 + pr) * b ** 3 * a ** 3
    return factor * (term1 + term2)

def build_constraints(inner_tris, outer_tris, full_mesh):
    cs = []
    cs = constraints.continuity_constraints(full_mesh[1], np.array([]), full_mesh[0])
    return constraints.sort_by_constrained_dof(cs)

def spherify(center, r, pts):
    D = scipy.spatial.distance.cdist(pts, center.reshape((1,center.shape[0])))
    return (r / D) * (pts - center) + center

def make_sphere(center, r, refinements):
    pts = np.array([[0,-r,0],[r,0,0],[0,0,r],[-r,0,0],[0,0,-r],[0,r,0]])
    tris = np.array([[1,0,2],[2,0,3],[3,0,4],[4,0,1],[5,1,2],[5,2,3],[5,3,4],[5,4,1]])
    m = pts, tris
    for i in range(refinements):
        m = mesh.refine(m)
        m = (spherify(center, r, m[0]), m[1])
    m = (m[0] + center, m[1])
    return m

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
    refine = 4
    a = 1.0
    b = 2.0
    sm = 1.0
    pr = 0.25
    ua = 2.0
    ub = 1.0

    flip_inner = False
    flip_outer = True

    m_inner = make_sphere(np.array([0,0,0]), a, refine)
    m_outer = make_sphere(np.array([0,0,0]), b, refine)
    if flip_outer:
        m_outer = mesh.flip_normals(m_outer)
    if flip_inner:
        m_inner = mesh.flip_normals(m_inner)

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

    input_nd = tri_pts / np.linalg.norm(tri_pts, axis = 2)[:,:,np.newaxis]
    input_nd[:n_inner] *= ua
    input_nd[n_inner:] *= ub

    input = input_nd.reshape(-1)

    selfop = MassOp(3, m[0], m[1]).mat

    t = Timer()
    eps = 0.01 * (2.0 ** -np.arange(10))
    Uop = SparseIntegralOp(
        eps, 20, 20, 7, 3, 6, 4.0,
        'U', sm, pr, m[0], m[1], use_tables = True, remove_sing = False
    )
    t.report('U')
    Top = SparseIntegralOp(
        eps, 20, 20, 10, 3, 6, 4.0,
        'T', sm, pr, m[0], m[1], use_tables = True, remove_sing = False
    )
    t.report('T')

    lhs = Uop
    rhs = Top.dot(input) + selfop.dot(input)

    soln = iterative_solve(lhs, cs, rhs)

    disp = np.array(soln).reshape((int(lhs.shape[0] / 9), 3, 3))
    disp_mag = np.sqrt(np.sum(disp ** 2, axis = 2))
    inner_disp_mag = disp_mag[:n_inner]
    outer_disp_mag = disp_mag[n_inner:]
    print(inner_disp_mag[:10])
    print(outer_disp_mag[:10])
    mean_inner = np.mean(inner_disp_mag)
    mean_outer = np.mean(outer_disp_mag)
    print(mean_inner, np.std(inner_disp_mag))
    print(mean_outer, np.std(outer_disp_mag))

    pa = mean_inner
    pb = mean_outer
    print(correct_displacement(a, b, pa, pb, sm, pr, np.array([a, b])))
    print(correct_displacement(a, b, -pa, pb, sm, pr, np.array([a, b])))
    print(correct_displacement(a, b, -pa, -pb, sm, pr, np.array([a, b])))
    print(correct_displacement(a, b, pa, -pb, sm, pr, np.array([a, b])))

def main():
    params = [False]
    outputs = [runner(p, do_solve = True) for p in params]
    for i, o in enumerate(outputs):
        np.save('debug/' + str(i) + '.npy', o)
    outputs = [np.load('debug/' + str(i) + '.npy') for i in range(len(params))]

if __name__ == '__main__':
    main()
