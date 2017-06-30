import numpy as np
from tectosaur.ops.dense_op import DenseOp
from tectosaur.ops.composite_op import CompositeOp
from tectosaur.ops.sum_op import SumOp

def test_composite_op():
    n = 20
    split_n = 10
    mat = np.random.rand(n, n)
    v = np.random.rand(n)
    comp_op = CompositeOp(
        (DenseOp(mat[:split_n,:split_n]), 0, 0),
        (DenseOp(mat[split_n:,:split_n]), split_n, 0),
        (DenseOp(mat[:split_n,split_n:]), 0, split_n),
        (DenseOp(mat[split_n:,split_n:]), split_n, split_n)
    )

    correct = mat.dot(v)
    check = comp_op.dot(v)
    np.testing.assert_almost_equal(correct, check)

def test_sum_op():
    mat = np.random.rand(2,2)
    v = np.random.rand(2)
    sum_op = SumOp([DenseOp(mat), DenseOp(mat)])
    np.testing.assert_almost_equal(sum_op.dot(v), 2 * mat.dot(v))

