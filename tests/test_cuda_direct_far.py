import time
import cppimport
import numpy as np

from tectosaur.ops.sparse_integral_op import farfield_pts_wrapper, get_gpu_module

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
    obs_pts = np.random.rand(n, 3)
    obs_ns = normalize(np.random.rand(n, 3))
    src_pts = np.random.rand(n, 3)
    src_ns = obs_ns
    weights = np.random.rand(n, 3)


    get_gpu_module() #preload module so it doesn't get counted in the runtime
    start = time.time()
    params = [1.0, 0.25]
    result = farfield_pts_wrapper(
        k_name, obs_pts, obs_ns, src_pts, src_ns, weights, params
    ).reshape((n, 3))
    runtime = time.time() - start
    if timeit:
        timing(n, runtime, k_name, flops)

    # if testit:
    #     correct = fmm.direct_eval(
    #         "elastic" + k_name, obs_pts, obs_ns, src_pts, src_ns, [1.0, 0.25]
    #     )
    #     correct = correct.reshape((n * 3, n * 3))
    #     correct = correct.dot(weights.reshape(n * 3)).reshape((n, 3))
    #     np.testing.assert_almost_equal(
    #         np.abs((result - correct) / correct),
    #         np.zeros_like(result), 2
    #     )

def test_U():
    run_kernel(1000, 'elasticU', 28, testit = True)

def test_T():
    run_kernel(1000, 'elasticT', 63, testit = True)

def test_A():
    run_kernel(1000, 'elasticA', 63, testit = True)

def test_H():
    run_kernel(1000, 'elasticH', 102, testit = True)

if __name__ == '__main__':
    run_kernel(128 * 512, 'elasticU', 28, timeit = True)
    run_kernel(128 * 512, 'elasticA', 63, timeit = True)
    run_kernel(128 * 512, 'elasticT', 63, timeit = True)
    run_kernel(128 * 512, 'elasticH', 102, timeit = True)
