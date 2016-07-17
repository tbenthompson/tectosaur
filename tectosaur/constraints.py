import scipy.sparse
import numpy as np
from tectosaur.adjacency import find_touching_pts
import tectosaur.geometry as geom

def check_if_crosses_fault(tri1, tri2, fault_touching_pts, fault_tris, pts):
    for fault_tri_idx,_ in fault_touching_pts:
        fault_t = fault_tris[fault_tri_idx]
        plane = pts[fault_t]
        tri1_sides = [geom.which_side_point(plane, pts[tri1[d]]) for d in range(3)]
        tri2_sides = [geom.which_side_point(plane, pts[tri2[d]]) for d in range(3)]
        side1 = geom.tri_side(tri1_sides)
        side2 = geom.tri_side(tri2_sides)
        if side1 != side2:
            return True
    return False

def constraints(surface_tris, fault_tris, pts):
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    touching_pt = find_touching_pts(surface_tris)
    if fault_tris.shape[0] > 0:
        fault_touching_pt = find_touching_pts(fault_tris)
    constraints = []
    for i, tpt in enumerate(touching_pt):

        tri1_idx = tpt[0][0]
        tri1 = surface_tris[tri1_idx]
        for dependent in tpt[1:]:
            tri2_idx = dependent[0]
            tri2 = surface_tris[tri2_idx]

            # Check for anything that touches across the fault.
            crosses = (
                fault_tris.shape[0] > 0
                and check_if_crosses_fault(
                    tri1, tri2, fault_touching_pt[i], fault_tris, pts
                )
            )
            if crosses:
                print("HI")
                continue

            for d in range(3):
                indepedent_dof = tri1_idx * 9 + tpt[0][1] * 3 + d
                dependent_dof = tri2_idx * 9 + dependent[1] * 3 + d
                if dependent_dof <= indepedent_dof:
                    continue
                constraints.append((
                    [(1.0, dependent_dof), (-1.0, indepedent_dof)], 0.0
                ))
    # assert(len(touching_pt) == surface_tris.size - len(constraints) / 3)

    # X component = 1
    # Y comp = Z comp = 0
    slip = [-1, 0, 0]
    for i in range(n_surf_tris, n_surf_tris + n_fault_tris):
        for b in range(3):
            for d in range(3):
                dof = i * 9 + b * 3 + d
                constraints.append(([(1.0, dof)], slip[d]))
    constraints = sorted(constraints, key = lambda x: x[0][0][1])
    return constraints

def lagrange_constraints(lhs, rhs, cs):
    c_start = lhs.shape[0] - len(cs)
    for i, c in enumerate(cs):
        idx1 = c_start + i
        rhs[idx1] = c[1]
        for dw in c[0]:
            coeff = dw[0]
            idx2 = dw[1]
            lhs[idx1, idx2] = coeff
            lhs[idx2, idx1] = coeff

from collections import namedtuple
IsolatedTermEQ = namedtuple('IsolatedTermEQ', 'lhs_dof terms const')

def isolate_term_on_lhs(c, entry_idx):
    lhs_dof = c[0][entry_idx][1]
    lhs_wt = c[0][entry_idx][0]

    divided_negated_terms = [
        (-t[0] / lhs_wt, t[1])
        for i, t in enumerate(c[0])
        if i != entry_idx
    ]
    divided_rhs = c[1] / lhs_wt
    return IsolatedTermEQ(lhs_dof, divided_negated_terms, divided_rhs)

def substitute(c_victim, entry_idx, c_in):
    mult_factor = c_victim[0][entry_idx][0]

    out_terms = [t for i, t in enumerate(c_victim[0]) if i != entry_idx]
    out_terms.extend([(mult_factor * t[0], t[1]) for t in c_in.terms])
    out_rhs = c_victim[1] - mult_factor * c_in.const
    return (out_terms, out_rhs)

def combine_terms(c):
    out_terms = dict()
    for t in c[0]:
        if t[1] in out_terms:
            out_terms[t[1]] = (t[0] + out_terms[t[1]][0], t[1])
        else:
            out_terms[t[1]] = t
    return ([v for v in out_terms.values()], c[1])

def filter_zero_terms(c):
    return ([t for t in c[0] if np.abs(t[0]) > 1e-15], c[1])

def last_dof_idx(c):
    return np.argmax([e[1] for e in c[0]])

def make_reduced(c, matrix):
    for i, t in enumerate(c[0]):
        if t[1] in matrix:
            c_subs = substitute(c, i, matrix[t[1]])
            c_filtered = filter_zero_terms(c_subs)
            return make_reduced(c_filtered, matrix)
    return c

def reduce_constraints(cs):
    sorted_cs = sorted(cs, key = last_dof_idx)
    lower_tri_cs = dict()
    for c in sorted_cs:
        c_combined = combine_terms(c)
        c_filtered = filter_zero_terms(c_combined)
        c_lower_tri = make_reduced(c_filtered, lower_tri_cs)

        if len(c_lower_tri[0]) == 0:
            continue

        ldi = last_dof_idx(c_lower_tri)
        separated = isolate_term_on_lhs(c_lower_tri, ldi)
        lower_tri_cs[separated.lhs_dof] = separated
    return lower_tri_cs

def to_matrix(reduced_cs, n_total_dofs):
    cm = scipy.sparse.dok_matrix((
        n_total_dofs, n_total_dofs - len(reduced_cs.keys())
    ))
    next_new_dof = 0
    new_dofs = dict()
    for i in range(n_total_dofs):
        if i in reduced_cs:
            for t in reduced_cs[i].terms:
                assert(t[1] in new_dofs) # Should be true b/c the
                # constraints have been lower triangularized.

                cm[i, new_dofs[t[1]]] = t[0]
            #TODO: handle rhs/inhomogeneous constraints
        else:
            cm[i, next_new_dof] = 1
            new_dofs[i] = next_new_dof
            next_new_dof += 1
    return cm

def build_constraint_matrix(cs, n_total_dofs):
    reduced_cs = reduce_constraints(cs)
    return to_matrix(reduced_cs, n_total_dofs)
