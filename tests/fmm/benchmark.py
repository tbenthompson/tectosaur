import sys
import numpy as np
import tectosaur.fmm.fmm as fmm
from tectosaur.util.test_decorators import golden_master

from tectosaur.util.timer import Timer
from tectosaur.farfield import farfield_pts_direct

import logging
logger = logging.getLogger(__name__)

# K_name = 'laplaceS2'
# tensor_dim = 1
# mac = 3.0
# order = 10

K_name = 'laplaceS3'
tensor_dim = 1
mac = 2.0
order = 100

# K_name = 'elasticH3'
# tensor_dim = 3
# mac = 3.0
# order = 100

pts_per_cell = order
float_type = np.float32
dim = int(K_name[-1])
params = [1.0, 0.25]

def random_data(N):
    pts = np.random.rand(N, dim)
    ns = np.random.rand(N, dim)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
    input = np.random.rand(N, tensor_dim)
    return pts, ns, input

def ellipsoid_pts(N):
    assert(dim == 3)
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
    if dim == 3:
        X,Y,Z = np.meshgrid(xs,xs,xs)
        pts = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    elif dim == 2:
        X,Y = np.meshgrid(xs,xs)
        pts = np.array([X.flatten(), Y.flatten()]).T

    ns = np.random.rand(N ** dim, dim)
    ns /= np.linalg.norm(ns, axis = 1)[:,np.newaxis]
    input = np.random.rand(N ** dim, tensor_dim)
    return pts.copy(), ns.copy(), input.copy()

def direct_runner(pts, ns, input):
    t = Timer()
    out_direct = farfield_pts_direct(K_name, pts, ns, pts, ns, input.flatten(), params, float_type)
    t.report('eval direct')
    return out_direct

def fmm_runner(pts, ns, input_vals):
    t = Timer()

    cfg = fmm.make_config(K_name, params, 1.1, mac, order, float_type)
    tree = fmm.make_tree(pts, cfg, pts_per_cell)
    t.report('build tree')

    orig_idxs = np.array(tree.orig_idxs)
    input_tree = input_vals.reshape((-1,tensor_dim))[orig_idxs,:].reshape(-1)
    mapped_ns = ns[orig_idxs]
    t.report('map input to tree space')

    fmm_obj = fmm.FMM(tree, mapped_ns, tree, mapped_ns, cfg)
    t.report('setup fmm')

    fmm.report_interactions(fmm_obj)
    t.report('report')

    evaluator = fmm.FMMEvaluator(fmm_obj)

    import asyncio
    output = asyncio.get_event_loop().run_until_complete(evaluator.eval(input_tree))
    t.report('eval fmm')

    # for i in range(2):
    #     output = evaluator.eval(input_tree)
    #     t.report('eval fmm')

    to_orig = fmm_obj.to_orig(output)
    t.report('map to input space')
    return to_orig

def check(A, B):
    L2B = np.sqrt(np.sum(B ** 2))
    L2Diff = np.sqrt(np.sum((A - B) ** 2))
    relL2 = L2Diff / L2B
    print(L2B, L2Diff, relL2)

if __name__ == '__main__':
    N = int(sys.argv[1])
    test = sys.argv[2] == 'test'
    np.random.seed(10)
    # N = 1000000
    # data = random_data(N)
    # N = 10000000
    # data = ellipsoid_pts(N)
    data = grid_data(int(N ** (1.0 / float(dim))))
    A = fmm_runner(*data).flatten()
    if test:
        B = direct_runner(*data)
        check(A, B)
