import numpy as np
import scipy.sparse
import tectosaur.mesh as mesh
from tectosaur.constraints import constraints
from tectosaur.dense_integral_op import DenseIntegralOperator

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

def test_rearrange_constraint_eq():
    eqtn = ([(3,0),(-1,1),(4,2)], 13.7)
    rearr = isolate_term_on_lhs(eqtn, 2)
    assert(rearr == (2, [(-3.0 / 4.0, 0), (1.0 / 4.0, 1)], 13.7 / 4.0))

def substitute(c_victim, entry_idx, c_in):
    mult_factor = c_victim[0][entry_idx][0]

    out_terms = [t for i, t in enumerate(c_victim[0]) if i != entry_idx]
    out_terms.extend([(mult_factor * t[0], t[1]) for t in c_in.terms])
    out_rhs = c_victim[1] - mult_factor * c_in.const
    return (out_terms, out_rhs)

def subs_test(victim, sub_in, correct):
    in_rearr = isolate_term_on_lhs(sub_in, 0)
    result = substitute(victim, 0, in_rearr)
    assert(result == correct)

def combine_terms(c):
    out_terms = dict()
    for t in c[0]:
        if t[1] in out_terms:
            out_terms[t[1]] = (t[0] + out_terms[t[1]][0], t[1])
        else:
            out_terms[t[1]] = t
    return ([v for v in out_terms.values()], c[1])

def test_combine_terms():
    assert(combine_terms(([(1, 1), (2, 1)], 0.0)) == ([(3,1)], 0.0))

def test_subs_rhs():
    eqtn0 = ([(1,1), (3,1)], 4.0)
    eqtn1 = ([(1,1)], 2.0)
    correct = ([(3,1)], 2.0)
    subs_test(eqtn0, eqtn1, correct)

def filter_zero_terms(c):
    return ([t for t in c[0] if np.abs(t[0]) > 1e-15], c[1])

def test_filter_zero():
    assert(filter_zero_terms(([(1, 0), (0, 1)], 0.0)) == ([(1, 0)], 0.0))

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

def test_constraint_matrix():
    cs = [([(1, 0), (-1, 1)], 0.0)]
    cm = build_constraint_matrix(cs, 3)
    assert(cm.shape == (3, 2))
    np.testing.assert_almost_equal(cm.todense(), [[1, 0], [1, 0], [0, 1]])

def test_constraint_matrix_harder():
    cs = [
        ([(1, 5), (-1, 1)], 0.0),
        ([(1, 3), (0.25, 0)], 0.0),
        ([(1, 2), (0.5, 3), (0.5, 4)], 0.0)
    ]
    cm = build_constraint_matrix(cs, 7)
    assert(cm.shape == (7, 4))
    correct = [
        [1,0,0,0],[0,1,0,0],[0,0,1,0], [-0.25,0,0,0],
        [0.25,0,-2,0],[0,1,0,0],[0,0,0,1]
    ]
    np.testing.assert_almost_equal(cm.todense(), correct)

sm = 1.0
pr = 0.25
w = 4
corners = [[w, w, 0], [w, -w, 0], [-w, -w, 0], [-w, w, 0]]
m = mesh.rect_surface(3, 3, corners)
cs = constraints(m[1], np.empty((0,3)), m[0])
cm = build_constraint_matrix(cs, m[1].shape[0] * 9)
cm = cm.todense()

old_iop = None
for nq in [5, 10, 15, 20, 25]:
    iop = DenseIntegralOperator(nq, nq, 8, 4, sm, pr, m[0], m[1])
    if nq == 5:
        old_iop = iop
        continue
    Khat = cm.T.dot(iop.mat.dot(cm))
    Khat_old = cm.T.dot(old_iop.mat.dot(cm))
    print(np.max(Khat_old - Khat))
    print(np.max(Khat_old))
    old_iop = iop
