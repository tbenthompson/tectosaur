from tectosaur.constraints import *
import numpy as np

def test_rearrange_constraint_eq():
    eqtn = ([(3,0),(-1,1),(4,2)], 13.7)
    rearr = isolate_term_on_lhs(eqtn, 2)
    assert(rearr == (2, [(-3.0 / 4.0, 0), (1.0 / 4.0, 1)], 13.7 / 4.0))

def subs_test(victim, sub_in, correct):
    in_rearr = isolate_term_on_lhs(sub_in, 0)
    result = substitute(victim, 0, in_rearr)
    assert(result == correct)

def test_combine_terms():
    assert(combine_terms(([(1, 1), (2, 1)], 0.0)) == ([(3,1)], 0.0))

def test_subs_rhs():
    eqtn0 = ([(1,1), (3,1)], 4.0)
    eqtn1 = ([(1,1)], 2.0)
    correct = ([(3,1)], 2.0)
    subs_test(eqtn0, eqtn1, correct)

def test_filter_zero():
    assert(filter_zero_terms(([(1, 0), (0, 1)], 0.0)) == ([(1, 0)], 0.0))

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

