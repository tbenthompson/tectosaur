import time
from math import factorial
import scipy.special
import scipy.spatial
import numpy as np

import tectosaur as tct
from tectosaur.mesh.modify import concat
from tectosaur.fmm.tsfmm import *
import tectosaur.util.gpu as gpu

def fmm_tester(K_name):
    order = 4
    float_type = np.float64
    quad_order = 2
    K_params = np.array([1.0, 0.25])

    n = 20
    offset = 0.0
    corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
    m_src = tct.make_rect(n, n, corners)
    v = np.random.rand(m_src[1].shape[0] * 9).astype(float_type)

    m_obs = tct.make_rect(n, n, corners)
    m_obs[0][:,0] += offset

    full_m = concat(m_src, m_obs)
    src_subset = np.arange(0, m_src[1].shape[0])
    obs_subset = np.arange(0, m_obs[1].shape[0]) + m_src[1].shape[0]
    op = tct.TriToTriDirectFarfieldOp(
        quad_order, K_name, K_params, full_m[0], full_m[1],
        float_type, obs_subset, src_subset
    )
    y1 = op.dot(v)

    fmm = TSFMM(
        m_obs, m_src, params = K_params, order = order,
        quad_order = quad_order, float_type = float_type,
        mac = 2.5, max_pts_per_cell = 20, n_workers_per_block = 128
    )
    report_interactions(fmm)

    y2 = fmm.dot(v)
    print(order, np.linalg.norm((y1 - y2)) / np.linalg.norm(y1))
    np.testing.assert_almost_equal(y1, y2)

def test_fmmU():
    fmm_tester('elasticU3')

def test_fmmT():
    fmm_tester('elasticT3')

def benchmark():
    compare = False
    np.random.seed(123456)
    float_type = np.float32
    n = 200
    corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
    m = tct.make_rect(n, n, corners)
    v = (100000 * np.random.rand(m[1].shape[0] * 9)).astype(float_type)
    t = tct.Timer()

    if compare:
        all_tris = np.arange(m[1].shape[0])
        op = tct.TriToTriDirectFarfieldOp(
            2, 'elasticU3', [1.0, 0.25], m[0], m[1],
            float_type, all_tris, all_tris
        )
        t.report('build direct')
        for i in range(2):
            y1 = op.dot(v)
            t.report('op.dot direct')

        all_tris = np.arange(m[1].shape[0])
        oldfmm = tct.FMMFarfieldOp(4.0, 400, 1e-5)(
            2, 'elasticU3', [1.0, 0.25], m[0], m[1],
            float_type, all_tris, all_tris
        )
        t.report('build oldfmm')
        for i in range(2):
            oldfmm.dot(v)
            t.report('op.dot oldfmm')

    # TODO: still maybe some room in p2p compared to direct
    # TODO
    # TODO: maybe do full fmm?
    fmm = TSFMM(
        m, m, params = np.array([1.0, 0.25]), order = 4,
        quad_order = 2, float_type = float_type,
        mac = 2.5, max_pts_per_cell = 80, n_workers_per_block = 128
    )
    report_interactions(fmm)
    t.report('build')
    out = fmm.dot(v)
    t.report('first dot')
    out = fmm.dot(v)
    t.report('second dot')
    for i in range(1):
        start = time.time()
        out = fmm.dot(v)
        t.report('third dot')
        took = time.time() - start
        interactions = m[1].shape[0] ** 2
        print('million rows/sec', m[1].shape[0] / took / 1e6)
        print('billion interactions/sec', interactions / took / 1e9)

    filename = 'tests/fmm/taylorbenchmarkcorrect.npy'
    # np.save(filename, out)
    correct = np.load(filename)
    # print(out, correct, y1)
    np.testing.assert_almost_equal(out, correct, 5)

if __name__ == "__main__":
    benchmark()
