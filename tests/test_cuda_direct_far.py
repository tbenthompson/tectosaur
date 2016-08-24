import cppimport
import numpy as np

import pycuda.driver as drv
from tectosaur.util.gpu import load_gpu

from test_decorators import golden_master

import cppimport
fmm = cppimport.imp("tectosaur._fmm._fmm")._fmm._fmm

def normalize(vs):
    return vs / np.linalg.norm(vs, axis = 1).reshape((vs.shape[0], 1))

def timing(n, runtime, name, flops):
    print("for " + name)
    cycles = runtime * 5e12
    entries = ((n * 3) ** 2)
    cycles_per_entry = cycles / entries
    print("total time: " + str(runtime))
    print("pts: " + str(n))
    print("interacts: " + str(entries))
    print("cycles/interact: " + str(cycles_per_entry))
    print("total flop count: " + str(flops * n ** 2))
    print("Tflop/s: " + str(flops * n ** 2 / runtime / 1e12))

def run_kernel(n, k_name, flops, testit = False, timeit = False):
    block_size = 128
    np.random.seed(100)
    obs_pts = np.random.rand(n, 3).astype(np.float32)
    obs_ns = normalize(np.random.rand(n, 3).astype(np.float32))
    src_pts = np.random.rand(n, 3).astype(np.float32)
    src_ns = obs_ns
    weights = np.random.rand(n, 3).astype(np.float32)

    gpu_module = load_gpu(
        'tectosaur/integrals.cu',
        tmpl_args = dict(block_size = block_size)
    )

    farfield = gpu_module.get_function("farfield_pts" + k_name)
    block = (block_size, 1, 1)
    grid = (int(np.ceil(n / block[0])), 1)
    result = np.zeros((n, 3)).astype(np.float32)
    runtime = farfield(
        drv.Out(result), drv.In(obs_pts), drv.In(obs_ns),
        drv.In(src_pts), drv.In(src_ns), drv.In(weights),
        np.float32(1.0), np.float32(0.25), np.int32(n), np.int32(n),
        block = block,
        grid = grid,
        time_kernel = True
    )
    if timeit:
        timing(n, runtime, k_name, flops)

    if testit:
        correct = fmm.direct_eval(
            "elastic" + k_name, obs_pts, obs_ns, src_pts, src_ns, [1.0, 0.25]
        )
        correct = correct.reshape((n * 3, n * 3))
        correct = correct.dot(weights.reshape(n * 3)).reshape((n, 3))
        np.testing.assert_almost_equal(
            np.abs((result - correct) / correct),
            np.zeros_like(result), 2
        )

def test_U():
    run_kernel(1000, 'U', 28, testit = True)

def test_T():
    run_kernel(1000, 'T', 63, testit = True)

def test_A():
    run_kernel(1000, 'A', 63, testit = True)

def test_H():
    run_kernel(1000, 'H', 102, testit = True)

if __name__ == '__main__':
    run_kernel(128 * 512, 'U', 28, timeit = True)
    run_kernel(128 * 512, 'A', 63, timeit = True)
    run_kernel(128 * 512, 'T', 63, timeit = True)
    run_kernel(128 * 512, 'H', 102, timeit = True)
