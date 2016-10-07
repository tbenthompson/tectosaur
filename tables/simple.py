from tectosaur.dense_integral_op import DenseIntegralOp
import numpy as np

pts = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype = np.float64)
tris = np.array([[0,1,2]])
eps = 0.1 * (2.0 ** -np.arange(3))
print(eps)
op = DenseIntegralOp(eps, 25, 15, 10, 3, 10, 3.0, 'H', 1.0, 0.25, pts, tris)
op3 = DenseIntegralOp(eps, 26, 15, 10, 3, 10, 3.0, 'H', 1.0, 0.25, pts, tris)
op2 = DenseIntegralOp(eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, 0.25, pts, tris, use_tables = True)
import ipdb; ipdb.set_trace()
