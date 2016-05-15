from tectosaur.tensors import *
from tectosaur.elastic import *
import numpy as np

def test_tensors():
    t = [[0, -1], [1, 0]]
    t2 = transpose(tensor_sum(tensor_mult(t, 2), tensor_negate(t)))
    np.testing.assert_almost_equal(t2, [[0, 1], [-1, 0]])

def test_sym_skw():
    t = [[3, 2], [1, 0]]
    sym_t = SYM(t)
    skw_t = SKW(t)
    np.testing.assert_almost_equal(sym_t, [[3, 1.5], [1.5, 0]])
    np.testing.assert_almost_equal(skw_t, [[0, 0.5], [-0.5, 0]])

def test_outer():
    np.testing.assert_almost_equal(tensor_outer([0, 1], [1, 2]), [[0, 0], [1, 2]])
