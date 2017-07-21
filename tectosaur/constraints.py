import scipy.sparse
import numpy as np
from collections import namedtuple

IsolatedTermEQ = namedtuple('IsolatedTermEQ', 'lhs_dof terms rhs')
Term = namedtuple('Term', 'val dof')
ConstraintEQ = namedtuple('ConstraintEQ', 'terms rhs')

def lagrange_constraints(lhs, rhs, cs):
    c_start = lhs.shape[0] - len(cs)
    for i, c in enumerate(cs):
        idx1 = c_start + i
        rhs[idx1] = c.rhs
        for t in c.terms:
            lhs[idx1, t.dof] = t.val
            lhs[t.dof, idx1] = t.val

def isolate_term_on_lhs(c, entry_idx):
    lhs = c.terms[entry_idx]

    divided_negated_terms = [
        Term(-t.val / lhs.val, t.dof)
        for i, t in enumerate(c.terms)
        if i != entry_idx
    ]
    divided_rhs = c[1] / lhs.val
    return IsolatedTermEQ(lhs.dof, divided_negated_terms, divided_rhs)

def substitute(c_victim, entry_idx, c_in):
    mult_factor = c_victim.terms[entry_idx].val

    out_terms = [t for i, t in enumerate(c_victim.terms) if i != entry_idx]
    out_terms.extend([Term(mult_factor * t.val, t.dof) for t in c_in.terms])
    out_rhs = c_victim.rhs - mult_factor * c_in.rhs
    return ConstraintEQ(out_terms, out_rhs)

def combine_terms(c):
    out_terms = dict()
    for t in c.terms:
        if t.dof in out_terms:
            out_terms[t.dof] = Term(t.val + out_terms[t.dof].val, t.dof)
        else:
            out_terms[t.dof] = t
    return ConstraintEQ([v for v in out_terms.values()], c.rhs)

def filter_zero_terms(c):
    return ConstraintEQ([t for t in c.terms if np.abs(t.val) > 1e-15], c.rhs)

def last_dof_idx(c):
    return np.argmax([t.dof for t in c.terms])

def max_dof(c):
    return max([t.dof for t in c.terms])

def make_reduced(c, matrix):
    for i, t in enumerate(c.terms):
        if t.dof in matrix:
            c_subs = substitute(c, i, matrix[t.dof])
            c_comb = combine_terms(c_subs)
            c_filtered = filter_zero_terms(c_comb)
            return make_reduced(c_filtered, matrix)
    return c

def reduce_constraints(cs, n_total_dofs):
    sorted_cs = sorted(cs, key = max_dof)
    lower_tri_cs = dict()
    for c in sorted_cs:
        c_combined = combine_terms(c)
        c_filtered = filter_zero_terms(c_combined)
        c_lower_tri = make_reduced(c_filtered, lower_tri_cs)

        if len(c_lower_tri.terms) == 0:
            # print("REDUNDANT CONSTRAINT")
            continue

        ldi = last_dof_idx(c_lower_tri)
        separated = isolate_term_on_lhs(c_lower_tri, ldi)
        lower_tri_cs[separated.lhs_dof] = separated

    # Re-reduce to catch anything that slipped through the cracks.
    for i in range(n_total_dofs):
        if i not in lower_tri_cs:
            continue
        lower_tri_cs[i] = make_reduced(lower_tri_cs[i], lower_tri_cs)

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
                assert(t.dof in new_dofs) # Should be true b/c the
                # constraints have been lower triangularized.

                cm[i, new_dofs[t.dof]] = t.val
        else:
            cm[i, next_new_dof] = 1
            new_dofs[i] = next_new_dof
            next_new_dof += 1
    return cm

def build_constraint_matrix(cs, n_total_dofs):
    reduced_cs = reduce_constraints(cs, n_total_dofs)
    mat = to_matrix(reduced_cs, n_total_dofs)
    rhs = np.zeros(mat.shape[0])
    for k, c in reduced_cs.items():
        rhs[k] = c.rhs
    return mat, rhs
