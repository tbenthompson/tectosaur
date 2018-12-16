import logging
import numpy as np

import tectosaur as tct
tct.logger.setLevel(logging.INFO)

n = 50
corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
m = tct.make_rect(n, n, corners)

t = tct.Timer()
all_tris = np.arange(m[1].shape[0])
op = tct.FMMFarfieldOp(mac = 4.5, pts_per_cell = 100)(
    2, 'elasticRT3', [1.0, 0.25], m[0], m[1], np.float32, all_tris, all_tris
)
t.report('build op')

x = np.random.rand(m[1].shape[0] * 9)
t.report('x')

for i in range(1):
    y = op.dot(x)
    t.report('op.dot(x)')
