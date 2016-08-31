from tectosaur.dense_taylor_integral_op import *
from tectosaur.dense_integral_op import DenseIntegralOp
from tectosaur.mesh import *

def test_full_op():
    m = rect_surface(2, 2, [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    last = 1.0
    for nq in range(10,200,5):
        print(nq)
        iop = DenseTaylorIntegralOp(
            0.01, 1, nq, 15, 10, 7, 3.0, 3,
            'H', 1e0, 0.25, m[0], m[1]
        )
        cur = iop.mat[0,0]
        print(cur, last - cur)
        eps = [0.2, 0.1, 0.05, 0.025, 0.0125, 0.0125 / 2.0]
        iop2 = DenseIntegralOp(eps, (1,1,nq,nq), 1, 6, 3, 6, 4.0, 'H', 1e0, 0.25, m[0], m[1])
        print(iop2.mat[0,0])
        last = cur
    print(7.24634826e-02)
    # import ipdb; ipdb.set_trace()
    # np.testing.assert_almost_equal(iop.mat, iop2.mat, 2)
