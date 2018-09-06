import time
import numpy as np

from tectosaur.farfield import farfield_pts_direct, get_gpu_module
from tectosaur.ops.sparse_farfield_op import TriToTriDirectFarfieldOp
from tectosaur.ops.sparse_farfield_op import PtToPtDirectFarfieldOp
from tectosaur.util.geometry import normalize
from tectosaur.mesh.mesh_gen import make_rect
from tectosaur.mesh.modify import concat

def make_meshes(n_m = 8, sep = 2, w = 1, n_m2 = None):
    if n_m2 is None:
        n_m2 = n_m

    m1 = make_rect(n_m, n_m, [
        [-w, 0, w], [-w, 0, -w],
        [w, 0, -w], [w, 0, w]
    ])
    m2 = make_rect(n_m2, n_m2, [
        [-w, sep, w], [-w, sep, -w],
        [w, sep, -w], [w, sep, w]
    ])
    m = concat(m1, m2)
    surf1_idxs = np.arange(m1[1].shape[0])
    surf2_idxs = (surf1_idxs[-1] + 1) + surf1_idxs
    return m, surf1_idxs, surf2_idxs

def test_tri_tri_farfield():
    m, surf1_idxs, surf2_idxs = make_meshes()
    T1, T2 = [
        C(
            2, 'elasticT3', [1.0,0.25], m[0], m[1],
            np.float32, obs_subset = surf1_idxs,
            src_subset = surf2_idxs
        ) for C in [PtToPtDirectFarfieldOp, TriToTriDirectFarfieldOp]
    ]
    in_vals = np.random.rand(T1.shape[1])
    out1 = T1.dot(in_vals)
    out2 = T2.dot(in_vals)
    np.testing.assert_almost_equal(out1, out2)

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
    weights = np.random.rand(n, 3).flatten()

    start = time.time()
    params = [1.0, 0.25]
    result = farfield_pts_direct(
        k_name, obs_pts, obs_ns, src_pts, src_ns, weights, params, np.float32
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
    run_kernel(1000, 'elasticU3', 28, testit = True)

def test_T():
    run_kernel(1000, 'elasticT3', 63, testit = True)

def test_A():
    run_kernel(1000, 'elasticA3', 63, testit = True)

def test_H():
    run_kernel(1000, 'elasticH3', 102, testit = True)

if __name__ == '__main__':
    n = 32 * 512
    run_kernel(n, 'elasticU3', 28, timeit = True)
    run_kernel(n, 'elasticA3', 63, timeit = True)
    run_kernel(n, 'elasticT3', 63, timeit = True)
    run_kernel(n, 'elasticH3', 102, timeit = True)
    run_kernel(n, 'laplaceS3', 3, timeit = True)
