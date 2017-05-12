import sys

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri

from tectosaur.fmm_integral_op import FMMIntegralOp
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

def traction_bie(sm, pr, m, input, solve_for, op_type):
    # Why am I not able to get the same accuracy with the traction BIE as I am
    # with the displacement BIE? Is there something wrong with how I calculate things?
    # Maybe with the traction inputs?
    selfop = MassOp(3, m[0], m[1])
    selfop.mat *= -1

    nqv = 15
    t = Timer()
    Hop = SparseIntegralOp(
        [], 1, 1, (nqv,nqv*2,nqv), 3, 6, 4.0,
        'H', sm, pr, m[0], m[1], use_tables = True
    )
    t.report('H')
    Aop = SparseIntegralOp(
        [], 1, 1, 7, 3, 6, 4.0,
        'A', sm, pr, m[0], m[1], use_tables = True
    )
    t.report('A')

    if solve_for is 't':
        lhs = SumOp([Aop, selfop])
        rhs = Hop.dot(input)
    else:
        lhs = Hop
        rhs = Aop.dot(input) + selfop.dot(input)
    return lhs, rhs

def displacement_bie(sm, pr, m, input, solve_for, op_type):
    selfop = MassOp(3, m[0], m[1])

    t = Timer()
    Uop = op_type(
        [], 1, 1, 5, 3, 4, 4.0,
        'U', sm, pr, m[0], m[1], use_tables = True
    )
    Uop2 = SparseIntegralOp(
        [], 1, 1, 5, 3, 4, 4.0,
        'U', sm, pr, m[0], m[1], use_tables = True
    )
    v = np.random.rand(Uop2.shape[1])
    a = Uop2.dot(v)
    b = Uop.dot(v)
    import ipdb; ipdb.set_trace()
    t.report('U')
    Top = SparseIntegralOp(
        [], 1, 1, 5, 3, 4, 4.0,
        'T', sm, pr, m[0], m[1], use_tables = True
    )
    t.report('T')

    if solve_for is 't':
        lhs = Uop
        rhs = Top.dot(input) + selfop.dot(input)
    else:
        lhs = SumOp([Top, selfop])
        rhs = Uop.dot(input)
    return lhs, rhs

def runner(solve_for, use_bie, refine, use_fmm):
    if use_fmm == 'fmm':
        op_type = FMMIntegralOp
    else:
        op_type = SparseIntegralOp

    a = 1.0
    b = 2.0
    sm = 1.0
    pr = 0.25
    pa = 3.0
    pb = 1.0
    center = [0, 0, 0]

    ua, ub = correct_displacement(a, b, pa, -pb, sm, pr, np.array([a, b]))
    print(correct_displacement(a, b, pa, -pb, sm, pr, np.array([a, b])))
    print(correct_displacement(a, b, -pa, -pb, sm, pr, np.array([a, b])))
    print(correct_displacement(a, b, -pa, pb, sm, pr, np.array([a, b])))
    print(correct_displacement(a, b, pa, pb, sm, pr, np.array([a, b])))

    t = Timer()
    m_inner = mesh.make_sphere(center, a, refine)
    m_outer = mesh.make_sphere(center, b, refine)
    m_outer = mesh.flip_normals(m_outer)
    t.report('make meshes')

    m = mesh.concat(m_inner, m_outer)
    t.report('concat meshes')
    print(m[1].shape)
    n_inner = m_inner[1].shape[0]
    inner_tris = m[1][:n_inner]
    outer_tris = m[1][n_inner:]
    tri_pts = m[0][m[1]]
    # mesh.plot_mesh3d(*m)

    cs = build_constraints(inner_tris, outer_tris, m, solve_for)
    t.report('build constraints')

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

    if use_bie == 'u':
        lhs, rhs = displacement_bie(sm, pr, m, input, solve_for, op_type)
    elif use_bie == 't':
        lhs, rhs = traction_bie(sm, pr, m, input, solve_for, op_type)

    # soln = direct_solve(lhs, cs, rhs)
    soln = iterative_solve(lhs, cs, rhs)

    soln = np.array(soln).reshape((int(lhs.shape[0] / 9), 3, 3))
    soln_mag = np.sqrt(np.sum(soln ** 2, axis = 2))
    inner_soln_mag = soln_mag[:n_inner]
    outer_soln_mag = soln_mag[n_inner:]
    mean_inner = np.mean(inner_soln_mag)
    mean_outer = np.mean(outer_soln_mag)
    print(np.std(inner_soln_mag))
    print(np.std(outer_soln_mag))

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
    runner(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])

if __name__ == '__main__':
    main()
