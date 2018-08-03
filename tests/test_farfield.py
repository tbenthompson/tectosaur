import time
import numpy as np

from tectosaur.farfield import farfield_pts_direct, get_gpu_module
from tectosaur.ops.sparse_farfield_op import TriToTriDirectFarfieldOp
from tectosaur.ops.sparse_farfield_op import PtToPtDirectFarfieldOp
from tectosaur.util.geometry import normalize
from tectosaur.mesh.mesh_gen import make_rect
from tectosaur.mesh.modify import concat

def make_meshes():
    n_m = 8
    sep = 2
    m1 = make_rect(n_m, n_m, [
        [-1, 0, 1], [-1, 0, -1],
        [1, 0, -1], [1, 0, 1]
    ])
    m2 = make_rect(n_m, n_m, [
        [-1, sep, 1], [-1, sep, -1],
        [1, sep, -1], [1, sep, 1]
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

def regularized_tester(K):
    full_K_name = f'elastic{K}3'
    full_RK_name = f'elasticR{K}3'
    m, surf1_idxs, surf2_idxs = make_meshes()
    T1, T2 = [
        C(
            2, K, [1.0,0.25], m[0], m[1],
            np.float32, obs_subset = surf1_idxs,
            src_subset = surf2_idxs
        ) for C, K in [
            (PtToPtDirectFarfieldOp, 'elasticT3'),
            (TriToTriDirectFarfieldOp, 'elasticRT3')
        ]
    ]

    dof_pts = m[0][m[1][surf2_idxs]]
    dof_pts[:,:,1] -= dof_pts[0,0,1]

    def gaussian(a, b, c, x):
        return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))
    dist = np.linalg.norm(dof_pts.reshape(-1,3), axis = 1)
    slip = np.zeros((dof_pts.shape[0] * 3, 3))
    for d in range(3):
        slip[:,d] = gaussian(0.1 * d, 0.0, 0.3, dist)

    should_plot = False
    if should_plot:
        import matplotlib.pyplot as plt
        pts_slip = np.zeros(m[0].shape[0])
        pts_slip[m[1][surf2_idxs]] = slip[:,0].reshape((-1,3))
        plt.tricontourf(m[0][:,0], m[0][:,2], m[1], np.log10(np.abs(pts_slip) + 1e-12))
        plt.colorbar()
        plt.show()

    slip_flat = slip.flatten()
    out1 = T1.dot(slip_flat)
    out2 = T2.dot(slip_flat)
    np.testing.assert_almost_equal(out1, out2, 6)

def test_regularized_T_farfield():
    regularized_tester('A')

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
