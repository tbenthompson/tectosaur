from tectosaur.dense_integral_op import DenseIntegralOp
import numpy as np

pts = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype = np.float64)
tris = np.array([[0,1,2]])
op = DenseIntegralOp([0.1], 15, 15, 10, 3, 10, 3.0, 'H', 1.0, 0.25, pts, tris)
print(op.mat.reshape((3,3,3,3))[0,:,0,:])
