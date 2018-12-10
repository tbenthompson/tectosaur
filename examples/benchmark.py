import logging
import numpy as np

import tectosaur as tct
tct.logger.setLevel(logging.INFO)

n = 20
corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
m = tct.make_rect(n, n, corners)

t = tct.Timer()
op = tct.RegularizedSparseIntegralOp(
    8, 8, 8, 2, 5, 2.5,
    'elasticRT3', 'elasticRT3', [1.0, 0.25], m[0], m[1],
    np.float32,
    # farfield_op_type = tct.TriToTriDirectFarfieldOp,
    farfield_op_type = tct.FMMFarfieldOp(mac = 4.5, pts_per_cell = 100)
)
t.report('build op')

x = np.random.rand(op.shape[1])
t.report('x')

y = op.dot(x)
t.report('op.dot(x)')
import ipdb
ipdb.set_trace()
