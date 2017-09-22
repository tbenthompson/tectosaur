from tectosaur.constraints import *
from tectosaur.constraint_builders import *
import tectosaur.mesh.mesh_gen as mesh_gen
import tectosaur.mesh.modify as mesh_modify
import numpy as np

from tectosaur.util.logging import setup_logger
logger = setup_logger(__name__)


def test_rearrange_constraint_eq():
    eqtn = ConstraintEQ([Term(3,0),Term(-1,1),Term(4,2)], 13.7)
    rearr = isolate_term_on_lhs(eqtn, 2)
    assert(rearr.lhs_dof == 2)
    assert(rearr.c.terms[0].val == -0.75)
    assert(rearr.c.terms[0].dof == 0)
    assert(rearr.c.terms[1].val == 0.25)
    assert(rearr.c.terms[1].dof == 1)
    assert(rearr.c.rhs == 13.7 / 4.0)

def subs_test(victim, sub_in, correct):
    in_rearr = isolate_term_on_lhs(sub_in, 0)
    result = substitute(victim, 0, in_rearr)
    assert(result == correct)

def test_subs_rhs():
    eqtn0 = ConstraintEQ([Term(1,1), Term(3,1)], 4.0)
    eqtn1 = ConstraintEQ([Term(1,1)], 2.0)
    correct = ConstraintEQ([Term(3,1)], 2.0)
    subs_test(eqtn0, eqtn1, correct)

def test_combine_terms():
    assert(combine_terms(ConstraintEQ([Term(1, 1), Term(2, 1)], 0.0)) ==
            ConstraintEQ([Term(3,1)], 0.0))

def test_filter_zero():
    assert(filter_zero_terms(ConstraintEQ([Term(1, 0), Term(0, 1)], 0.0)) ==
        ConstraintEQ([Term(1, 0)], 0.0))


def test_constraint_matrix():
    cs = [ConstraintEQ([Term(1, 0), Term(-1, 1)], 0.0)]
    cm,rhs = build_constraint_matrix(cs, 3)
    assert(cm.shape == (3, 2))
    np.testing.assert_almost_equal(cm.todense(), [[1, 0], [1, 0], [0, 1]])

def test_constraint_matrix_harder():
    cs = [
        ConstraintEQ([Term(1, 5), Term(-1, 1)], 0.0),
        ConstraintEQ([Term(1, 3), Term(0.25, 0)], 0.0),
        ConstraintEQ([Term(1, 2), Term(0.5, 3), Term(0.5, 4)], 0.0)
    ]
    cm,rhs = build_constraint_matrix(cs, 7)
    assert(cm.shape == (7, 4))
    correct = [
        [1,0,0,0],[0,1,0,0],[0,0,1,0], [-0.25,0,0,0],
        [0.25,0,-2,0],[0,1,0,0],[0,0,0,1]
    ]
    np.testing.assert_almost_equal(cm.todense(), correct)

def test_constraint_matrix_rhs():
    cs = [
        ConstraintEQ([Term(1, 5), Term(-1, 1)], 0.0),
        ConstraintEQ([Term(1, 3), Term(0.25, 0)], 1.0),
        ConstraintEQ([Term(1, 2), Term(0.5, 3), Term(0.5, 4)], 2.0)
    ]
    cm, rhs = build_constraint_matrix(cs, 7)
    np.testing.assert_almost_equal(rhs, [0,0,0,1.0,3.0,0,0])

def test_constraint_double():
    cs = [
        ConstraintEQ([Term(1, 0), Term(1, 1), Term(1, 2)], 0.0),
        ConstraintEQ([Term(1, 0), Term(-1, 1), Term(1, 2)], 0.0),
    ]
    cm, rhs = build_constraint_matrix(cs, 3)
    np.testing.assert_almost_equal(cm.todense(), np.array([[1, 0, -1]]).T)

def test_constraint_triple():
    cs = [
        ConstraintEQ([Term(1, 0), Term(1, 1), Term(1, 2), Term(1, 3)], 0.0),
        ConstraintEQ([Term(1, 0), Term(-1, 1), Term(1, 2), Term(1, 3)], 0.0),
        ConstraintEQ([Term(1, 0), Term(1, 1), Term(-1, 2), Term(1, 3)], 0.0),
    ]
    cm, rhs = build_constraint_matrix(cs, 4)
    np.testing.assert_almost_equal(cm.todense(), np.array([[1, 0, 0, -1]]).T)

def test_find_free_edges():
    tris = np.array([[0,1,2],[2,1,3]])
    free_es = find_free_edges(tris)
    assert(len(free_es) == 4)
    for e in [(0,0), (0,2), (1,1), (1,2)]:
        assert(e in free_es)

def test_free_edge_constraints():
    cs = free_edge_constraints([[0,1,2],[0,2,3],[0,3,4],[0,4,1]])
    dofs = [c.terms[0].dof for c in cs]
    assert(0 not in dofs)
    assert(len(dofs) == 8 * 3)

def test_composite():
    cs1 = [ConstraintEQ([Term(1, 0)], 2)]
    cs2 = [ConstraintEQ([Term(1, 0)], 3)]
    cs = build_composite_constraints((cs1, 2), (cs2, 3))
    assert(cs[0].terms[0].dof == 2)
    assert(cs[0].rhs == 2)
    assert(cs[1].terms[0].dof == 3)
    assert(cs[1].rhs == 3)

def test_redundant_continuity():
    n = 13
    corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
    m = mesh_gen.make_rect(n, n, corners)
    cs = continuity_constraints(m[1], np.array([]), m[0])
    n_total_dofs = m[1].size * 3
    rows, cols, vals, rhs, n_unique_cs = fast_constraints.build_constraint_matrix(cs, n_total_dofs)
    n_rows = n_total_dofs
    n_cols = n_total_dofs - n_unique_cs
    cm = scipy.sparse.csr_matrix((vals, (rows, cols)), shape = (n_rows, n_cols))
    assert(cm.shape[1] == 3 * n ** 2)

def test_faulted_continuity():
    n = 3
    corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
    m = mesh_gen.make_rect(n, n, corners)
    fault_corners = [[-1.0, 0.0, 0.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    m2 = mesh_gen.make_rect(n, n, fault_corners)
    all_mesh = mesh_modify.concat(m, m2)
    surface_tris = all_mesh[1][:m[1].shape[0]]
    fault_tris = all_mesh[1][m[1].shape[0]:]

    cs = continuity_constraints(surface_tris, fault_tris, all_mesh[0])
    n_total_dofs = m[1].size * 3
    rows, cols, vals, rhs, n_unique_cs = fast_constraints.build_constraint_matrix(cs, n_total_dofs)
    n_rows = n_total_dofs
    n_cols = n_total_dofs - n_unique_cs
    cm = scipy.sparse.csr_matrix((vals, (rows, cols)), shape = (n_rows, n_cols))
    assert(cm.shape[1] == 36)

def benchmark_build_constraint_matrix():
    from tectosaur.util.timer import Timer
    from tectosaur.constraints import fast_constraints
    import scipy.sparse
    t = Timer()
    corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
    n = 100
    m = mesh_gen.make_rect(n, n, corners)
    t.report('make mesh')
    cs = continuity_constraints(m[1], np.array([]), m[0])
    t.report('make constraints')
    n_total_dofs = m[1].size * 3
    rows, cols, vals, rhs, n_unique_cs = fast_constraints.build_constraint_matrix(cs, n_total_dofs)
    t.report('build matrix')
    n_rows = n_total_dofs
    n_cols = n_total_dofs - n_unique_cs
    cm = scipy.sparse.csr_matrix((vals, (rows, cols)), shape = (n_rows, n_cols))
    t.report('to csr')

if __name__ == '__main__':
    benchmark_build_constraint_matrix()
