import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri

import tectosaur.mesh as mesh
from tectosaur.mass_op import MassOp
from tectosaur.sparse_integral_op import SparseIntegralOp
from tectosaur.sum_op import SumOp
import tectosaur.constraints as constraints

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

def build_constraints(inner_tris, outer_tris, full_mesh, solve_for):
    cs = []
    cs = constraints.continuity_constraints(full_mesh[1], np.array([]), full_mesh[0])
    # if solve_for == 'u':
    #     cs.extend(constraints.elastic_rigid_body_constraints(
    #         full_mesh[0], full_mesh[1],
    #         [[0, 0], [0, 1], [0, 2]]
    #     ))
    return cs

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
    solve_for = 't'

    refine = 4
    a = 1.0
    b = 2.0
    sm = 1.0
    pr = 0.25
    pa = 1.0
    pb = 2.0
    center = [5, 0, 0]

    ua, ub = correct_displacement(a, b, pa, -pb, sm, pr, np.array([a, b]))

    m_inner = mesh.make_sphere(center, a, refine)
    m_outer = mesh.make_sphere(center, b, refine)
    m_outer = mesh.flip_normals(m_outer)

    m = mesh.concat(m_inner, m_outer)
    print(m[1].shape)
    n_inner = m_inner[1].shape[0]
    inner_tris = m[1][:n_inner]
    outer_tris = m[1][n_inner:]
    tri_pts = m[0][m[1]]
    # mesh.plot_mesh3d(*m)

    cs = build_constraints(inner_tris, outer_tris, m, solve_for)

    # solving: u(x) + int(T*u) = int(U*t)
    # values in radial direction because the sphere is centered at (0,0,0)

    r = tri_pts - np.array(center)[np.newaxis, np.newaxis, :]
    input_nd = r / np.linalg.norm(r, axis = 2)[:,:,np.newaxis]
    if solve_for is 't':
        input_nd[:n_inner] *= ua
        input_nd[n_inner:] *= ub
    else:
        input_nd[:n_inner] *= pa
        input_nd[n_inner:] *= pb

    input = input_nd.reshape(-1)

    selfop = MassOp(3, m[0], m[1])

    t = Timer()
    Hop = SparseIntegralOp(
        [], 1, 1, 7, 3, 6, 4.0,
        'H', sm, pr, m[0], m[1], use_tables = True, remove_sing = True
    )
    t.report('H')
    Aop = SparseIntegralOp(
        [], 1, 1, 7, 3, 6, 4.0,
        'A', sm, pr, m[0], m[1], use_tables = True, remove_sing = False
    )
    t.report('A')

    selfop.mat *= 1
    if solve_for is 't':
        lhs = SumOp([Aop, selfop])
        rhs = Hop.dot(input)
    else:
        lhs = Hop
        rhs = Aop.dot(input) + selfop.dot(input)
    # t = Timer()
    # Uop = SparseIntegralOp(
    #     [], 1, 1, 7, 3, 6, 4.0,
    #     'U', sm, pr, m[0], m[1], use_tables = True, remove_sing = False
    # )
    # t.report('U')
    # Top = SparseIntegralOp(
    #     [], 1, 1, 7, 3, 6, 4.0,
    #     'T', sm, pr, m[0], m[1], use_tables = True, remove_sing = False
    # )
    # t.report('T')

    # if solve_for is 't':
    #     lhs = Uop
    #     rhs = Top.dot(input) + selfop.dot(input)
    # else:
    #     lhs = SumOp([Top, selfop])
    #     rhs = Uop.dot(input)

    soln = iterative_solve(lhs, cs, rhs)

    soln = np.array(soln).reshape((int(lhs.shape[0] / 9), 3, 3))
    soln_mag = np.sqrt(np.sum(soln ** 2, axis = 2))
    inner_soln_mag = soln_mag[:n_inner]
    outer_soln_mag = soln_mag[n_inner:]
    mean_inner = np.mean(inner_soln_mag)
    mean_outer = np.mean(outer_soln_mag)

    if solve_for is 't':
        print(mean_inner, pa)
        print(mean_outer, pb)
        np.testing.assert_almost_equal(mean_inner, pa, 1)
        np.testing.assert_almost_equal(mean_outer, pb, 1)
    else:
        print(mean_inner, ua)
        print(mean_outer, ub)
        np.testing.assert_almost_equal(mean_inner, ua, 1)
        np.testing.assert_almost_equal(mean_outer, ub, 1)

def main():
    params = [False]
    outputs = [runner(p, do_solve = True) for p in params]
    for i, o in enumerate(outputs):
        np.save('debug/' + str(i) + '.npy', o)
    outputs = [np.load('debug/' + str(i) + '.npy') for i in range(len(params))]

if __name__ == '__main__':
    main()
