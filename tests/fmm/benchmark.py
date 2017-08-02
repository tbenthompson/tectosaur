import numpy as np
from tectosaur.util.logging import setup_logger
import tectosaur.fmm.fmm_wrapper as fmm
from tectosaur.util.test_decorators import golden_master

from tectosaur.util.timer import Timer
from tectosaur.farfield import farfield_pts_direct

setup_logger(__name__)

K = 'laplaceS3'
tensor_dim = 1
mac = 2.0
order = 100

K = 'elasticT3'
tensor_dim = 3
mac = 3.0
order = 100

params = [1.0, 0.25]

def random_data(N):
    pts = np.random.rand(N, 3)
    ns = np.random.rand(N, 3)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
    input = np.random.rand(N, tensor_dim)
    return pts, ns, input

def ellipsoid_pts(N):
    a = 4.0
    b = 1.0
    c = 1.0
    uv = np.random.rand(N, 2)
    uv[:, 0] = (uv[:, 0] * np.pi) - np.pi / 2
    uv[:, 1] = (uv[:, 1] * 2 * np.pi) - np.pi
    x = a * np.cos(uv[:, 0]) * np.cos(uv[:, 1])
    y = b * np.cos(uv[:, 0]) * np.sin(uv[:, 1])
    z = c * np.sin(uv[:, 0])
    ns = np.random.rand(N, 3)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
    input = np.random.rand(N, tensor_dim)
    return np.array([x, y, z]).T.copy(), ns.copy(), input

def grid_data(N):
    xs = np.linspace(-1.0, 1.0, N)
    X,Y,Z = np.meshgrid(xs,xs,xs)
    pts = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    ns = np.random.rand(N * N * N, 3)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
    input = np.random.rand(N * N * N, tensor_dim)
    np.set_printoptions(precision=18)
    return pts.copy(), ns.copy(), input.copy()

def direct_runner(pts, ns, input):
    t = Timer()
    out_direct = farfield_pts_direct(K, pts, ns, pts, ns, input.flatten(), params)
    t.report('eval direct')
    return out_direct

def fmm_runner(pts, ns, input):
    t = Timer()

    pts_per_cell = 300

    tree = fmm.three.Octree(pts, pts_per_cell)
    t.report('build tree')

    orig_idxs = np.array(tree.orig_idxs)
    input_tree = input.reshape((-1,tensor_dim))[orig_idxs,:].reshape(-1)
    t.report('map input to tree space')

    mapped_ns = ns[orig_idxs]
    fmm_mat = fmm.three.fmmmmmmm(
        tree, mapped_ns, tree, mapped_ns, fmm.three.FMMConfig(1.1, mac, order, K, params)
    )
    t.report('setup fmm')
    fmm.report_interactions(fmm_mat)
    t.report('report')

    gpu_data = fmm.data_to_gpu(fmm_mat)
    t.report('data to gpu')

    output = fmm.eval_ocl(fmm_mat, input_tree, gpu_data)
    t.report('eval fmm')

    output = output.reshape((-1, tensor_dim))
    to_orig = np.empty_like(output)
    to_orig[orig_idxs,:] = output
    t.report('map to input space')
    return to_orig

def check(A, B):
    L2B = np.sqrt(np.sum(B ** 2))
    L2Diff = np.sqrt(np.sum((A - B) ** 2))
    relL2 = L2Diff / L2B
    print(L2B, L2Diff, relL2)

if __name__ == '__main__':
    np.random.seed(10)
    # N = 1000000
    # data = random_data(N)
    # N = 10000000
    # data = ellipsoid_pts(N)
    N = int(1e6 ** (1.0 / 3.0))
    data = grid_data(N)
    A = fmm_runner(*data).flatten()
    # B = direct_runner(*data)
    # check(A, B)
