import numpy as np
import time

from tectosaur.interpolate import barycentric_evalnd
from tectosaur.table_lookup import coincident_table

from test_interpolate import ptswts3d

from tectosaur.test_decorators import golden_master

import cppimport
fast_lookup = cppimport.imp("tectosaur.fast_lookup").fast_lookup

@golden_master
def test_coincident_lookup_single():
    pts = np.array([[0,0,0],[1,0,0],[0,1,0]])
    tris = np.array([[0,1,2]] * 100)
    start = time.time()
    res = coincident_table('H', 1.0, 0.25, pts, tris, True)
    end = time.time()
    print(end - start)
    return res

def test_fast_interp():
    pts, wts = ptswts3d(10)
    f = lambda xs: np.sin(xs[:,0] ** 3 - xs[:,1] * xs[:,2])
    vals = f(pts)[:,np.newaxis]
    xhat = np.array([[0.33, -0.1, 0.5]])
    res = barycentric_evalnd(pts, wts, vals, xhat)
    start = time.time()
    res2 = fast_lookup.barycentric_evalnd(pts, wts, vals, xhat)
    np.testing.assert_almost_equal(res, res2[0], 14)

