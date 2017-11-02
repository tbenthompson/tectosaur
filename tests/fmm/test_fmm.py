import sys
import numpy as np

import taskloaf as tsk
from tectosaur.farfield import farfield_pts_direct
from tectosaur.util.timer import Timer
import tectosaur.fmm.fmm as fmm
from tectosaur.fmm.c2e import direct_matrix

from dimension import dim

import logging
logger = logging.getLogger(__name__)

float_type = np.float64

def rand_pts(dim):
    def f(n, source):
        return np.random.rand(n, dim)
    return f

def ellipsoid_pts(n, source):
    a = 4.0
    b = 1.0
    c = 1.0
    uv = np.random.rand(n, 2)
    uv[:, 0] = (uv[:, 0] * np.pi) - np.pi / 2
    uv[:, 1] = (uv[:, 1] * 2 * np.pi) - np.pi
    x = a * np.cos(uv[:, 0]) * np.cos(uv[:, 1])
    y = b * np.cos(uv[:, 0]) * np.sin(uv[:, 1])
    z = c * np.sin(uv[:, 0])
    return np.array([x, y, z]).T

def m2l_test_pts(dim):
    def f(n, is_source):
        out = np.random.rand(n, dim)
        if is_source:
            out += 3.5
        return out
    return f

def get_pts(pts_builder, n):
    obs_pts = pts_builder(n, False)
    obs_ns = pts_builder(n, False)
    obs_ns /= np.linalg.norm(obs_ns, axis = 1)[:,np.newaxis]
    src_pts = pts_builder(n + 1, True)
    src_ns = pts_builder(n + 1, True)
    src_ns /= np.linalg.norm(src_ns, axis = 1)[:,np.newaxis]
    return obs_pts, obs_ns, src_pts, src_ns

def run_full(n, pts_builder, mac, order, kernel, params, max_pts_per_cell = None):
    if max_pts_per_cell is None:
        max_pts_per_cell = order
    t = Timer()

    obs_pts, obs_ns, src_pts, src_ns = get_pts(pts_builder, n)
    t.report('gen random data')

    dim = obs_pts.shape[1]

    cfg = fmm.make_config(kernel, params, 1.1, mac, order, float_type)

    obs_kd = fmm.make_tree(obs_pts, cfg, max_pts_per_cell)
    src_kd = fmm.make_tree(src_pts, cfg, max_pts_per_cell)
    obs_ns_kd = obs_ns[obs_kd.orig_idxs]
    src_ns_kd = src_ns[src_kd.orig_idxs]
    t.report('build trees')

    fmm_obj = fmm.FMM(obs_kd, obs_ns_kd, src_kd, src_ns_kd, cfg)
    t.report('setup fmm')

    evaluator = fmm.FMMEvaluator(fmm_obj)
    est = tsk.run(evaluator.eval(np.ones(fmm_obj.n_input)))
    t.report('eval fmm')

    return (
        np.array(obs_kd.pts), obs_ns_kd,
        np.array(src_kd.pts), src_ns_kd, est
    )

def check(est, correct, accuracy):
    rmse = np.sqrt(np.mean((est - correct) ** 2))
    rms_c = np.sqrt(np.mean(correct ** 2))
    logger.debug("L2ERR: " + str(rmse / rms_c))
    logger.debug("MEANERR: " + str(np.mean(np.abs(est - correct)) / rms_c))
    logger.debug("MAXERR: " + str(np.max(np.abs(est - correct)) / rms_c))
    logger.debug("MEANRELERR: " + str(np.mean(np.abs((est - correct) / correct))))
    logger.debug("MAXRELERR: " + str(np.max(np.abs((est - correct) / correct))))
    lhs = est / rms_c
    rhs = correct / rms_c
    np.testing.assert_almost_equal(lhs, rhs, accuracy)

def check_kernel(K, obs_pts, obs_ns, src_pts, src_ns, est, accuracy = 3):
    dim = obs_pts.shape[1]
    tensor_dim = int(est.size / obs_pts.shape[0])
    vec = np.ones(src_pts.shape[0] * tensor_dim)
    correct = farfield_pts_direct(
        K, obs_pts, obs_ns, src_pts, src_ns, vec, [1.0, 0.25], float_type
    )
    check(est, correct, accuracy)

def test_ones(dim):
    K = 'one' + str(dim)
    obs_pts, _, src_pts, _, est = run_full(5000, rand_pts(dim), 0.5, 1, K, [])
    assert(np.all(np.abs(est - np.array(src_pts, copy = False).shape[0]) < 1e-3))

import pytest
@pytest.fixture(params = ["laplaceS", "laplaceD", "laplaceH"])
def laplace_kernel(request):
    return request.param

def test_p2p(laplace_kernel, dim):
    K = laplace_kernel + str(dim)
    check_kernel(K, *run_full(
        1000, rand_pts(dim), 2.6, 1, K, [], max_pts_per_cell = 100000,
    ), accuracy = 10)

def test_laplace_all(laplace_kernel, dim):
    K = laplace_kernel + str(dim)
    np.random.seed(10)
    order = 16 if dim == 2 else 64
    check_kernel(K, *run_full(
        10000, rand_pts(dim), 2.6, order, K, []
    ), accuracy = 1)

def test_elastic():
    np.random.seed(10)
    dim = 3
    K = 'elasticU' + str(dim)
    order = 16 if dim == 2 else 64
    check_kernel(K, *run_full(
        4000, rand_pts(dim), 2.6, order, K, [1.0, 0.25]
    ), accuracy = 1)

def test_m2l(laplace_kernel, dim):
    K = laplace_kernel + str(dim)
    order = 15 if dim == 2 else 100
    check_kernel(K, *run_full(
        10000, m2l_test_pts(dim), 2.6, order, K, [], max_pts_per_cell = 100000
    ), accuracy = 3)

def test_irregular():
    K = "laplaceS3"
    check_kernel(K, *run_full(10000, ellipsoid_pts, 2.6, 35, K, []))

def test_direct_matrix():
    np.random.seed(10)
    K_name = "elasticU3"
    obs_pts, obs_ns, src_pts, src_ns = get_pts(rand_pts(3), 100)
    params = np.array([1.0, 0.25])
    cfg = fmm.make_config(K_name, params, 1.0, 1.0, 1, float_type)
    matrix = direct_matrix(cfg.gpu_module, cfg.K, obs_pts, obs_ns, src_pts, src_ns, params, float_type)
    est = matrix.dot(np.ones(matrix.shape[1]))
    check_kernel(K_name, obs_pts, obs_ns, src_pts, src_ns, est)

if __name__ == '__main__':
    test_ones(3)
