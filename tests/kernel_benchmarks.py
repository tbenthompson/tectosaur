import cppimport
import numpy as np

import pycuda.driver as drv
from tectosaur.util.gpu import load_gpu

import cppimport
fmm = cppimport.imp("tectosaur.fmm.fmm").fmm.fmm

def normalize(vs):
    return vs / np.linalg.norm(vs, axis = 1).reshape((vs.shape[0], 1))

n = 2048 * 32
block_size = 128
np.random.seed(100)
obs_pts = np.random.rand(n, 3).astype(np.float32)
obs_ns = normalize(np.random.rand(n, 3).astype(np.float32))
src_pts = np.random.rand(n, 3).astype(np.float32)
src_ns = obs_ns
weights = np.random.rand(n, 3).astype(np.float32)
gpu_module = load_gpu(
    'tectosaur/integrals.cu', print_code = True,
    tmpl_args = dict(block_size = block_size)
)

def check(k_name, result):
    return
    # correct = fmm.direct_eval(
    #     "elastic" + k_name, obs_pts, obs_ns, src_pts, src_ns, [1.0, 0.25]
    # )
    # correct = correct.reshape((n * 3, n * 3))
    # correct = correct.dot(weights.reshape(n * 3)).reshape((n, 3))
    # np.save('correct' + str(k_name) + '.npy', correct)
    correct = np.load('correct' + str(k_name) + '.npy')
    np.testing.assert_almost_equal(np.abs((result - correct) / correct), np.zeros_like(result), 2)

def timing(runtime, name, flops):
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

def run_kernel(k_name, flops):
    farfield = gpu_module.get_function("farfield_pts" + k_name)
    block = (block_size, 1, 1)
    grid = (int(n / block[0]), 1)
    result = np.zeros((n, 3)).astype(np.float32)
    runtime = farfield(
        drv.Out(result), drv.In(obs_pts), drv.In(obs_ns),
        drv.In(src_pts), drv.In(src_ns), drv.In(weights),
        np.float32(1.0), np.float32(0.25), np.int32(n),
        block = block,
        grid = grid,
        time_kernel = True
    )
    timing(runtime, k_name, flops)
    check(k_name, result)

if __name__ == '__main__':
    run_kernel('U', 28)
    run_kernel('T', 63)
    run_kernel('A', 63)
    run_kernel('H', 102)
